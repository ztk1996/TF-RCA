from re import T
import torch
import json
import argparse
import numpy as np
import random
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import time
import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score
from DataPreprocess.STVProcess import embedding_to_vector, load_dataset, process_one_trace
from DenStream.DenStream import DenStream
from CEDAS.CEDAS import CEDAS
from MicroRank.preprocess_data import get_span, get_service_operation_list, get_operation_slo
from MicroRank.online_rca import rca
from DataPreprocess.params import chaos_dict
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K = 3
start_str = '2022-01-13 00:00:00'
window_duration = 5 * 60 * 1000 # ms
AD_method = 'DenStream_withscore'    # 'DenStream_withscore', 'DenStream_withoutscore', 'CEDAS_withscore', 'CEDAS_withoutscore'


def timestamp(datetime: str) -> int:
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(timeArray)) * 1000
    return ts

def ms2str(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ms/1000))

def main():
    # ========================================
    # Init path vector encoder
    # ========================================
    all_path = []

    # ========================================
    # Create cluster object
    # ========================================
    if AD_method in ['DenStream_withscore', 'DenStream_withoutscore']:
        denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11)    # eps=0.3
    elif AD_method in ['CEDAS_withscore', 'CEDAS_withoutscore']:
        cedas = CEDAS(r0=0.2, decay=0.001, threshold=5)
        first_tag = True

    # ========================================
    # Init time window
    # ========================================
    start = timestamp(start_str)
    end = start + window_duration

    # ========================================
    # Init evaluation for AD
    # ========================================
    a_true, a_pred = [], []

    # ========================================
    # Init evaluation for RCA
    # ========================================
    r_true, r_pred = [], []    

    print('Start !')
    # main loop start
    while True:
        print('--------------------------------')
        print(f'time window: {ms2str(start)} ~ {ms2str(end)}')
        abnormal_count = 0
        abnormal_map = {}
        STV_map = {}
        label_map = {}
        tid_list = []
        dataset = load_dataset(start, end)
        if len(dataset) == 0:
            break
        
        a_true, a_pred = [], []
        # Init manual count
        manual_count = 0
        for _, data in tqdm(enumerate(dataset), desc="Time Window Samples: "):
            # ========================================
            # Path vector encoder
            # ========================================
            all_path = process_one_trace(data, all_path)
            STVector = embedding_to_vector(data, all_path)
            STV_map[data['trace_id']] = np.array(STVector)

            a_true.append(data['trace_bool'])

            if AD_method in ['DenStream_withoutscore', 'DenStream_withscore']:
                sample_label, label_status = denstream.Cluster_AnomalyDetector(np.array(STVector), data)
            elif AD_method in ['CEDAS_withoutscore', 'CEDAS_withscore']:
                if first_tag:
                    # 1. Initialization
                    sample_label, label_status = cedas.initialization(np.array(STVector), data)
                    first_tag = False
                else:
                    cedas.changed_cluster = None
                    # 2. Update Micro-Clusters
                    sample_label, label_status = cedas.Cluster_AnomalyDetector(np.array(STVector), data)
                    # 3. Kill Clusters
                    cedas.kill()
                    if cedas.changed_cluster and cedas.changed_cluster.count > cedas.threshold:
                        # 4. Update Cluster Graph
                        cedas.update_graph()

            # trace_id list
            tid = data['trace_id']
            tid_list.append(tid)

            if AD_method in ['DenStream_withoutscore', 'CEDAS_withoutscore']:
                # sample_label
                if sample_label == 'abnormal':
                    a_pred.append(1)
                    abnormal_map[tid] = True
                    abnormal_count += 1
                else:
                    abnormal_map[tid] = False
                    a_pred.append(0)

            # label_status
            if label_status == 'manual':
                manual_count += 1

        if AD_method == 'DenStream_withscore':
            labels, confidenceScores = denstream.get_labels_and_confidenceScores(STV_map)
            # sample_label
            for tid, sample_label in labels.items():
                if sample_label == 'abnormal':
                    a_pred.append(1)
                    abnormal_map[tid] = True
                    abnormal_count += 1
                else:
                    abnormal_map[tid] = False
                    a_pred.append(0)
        elif AD_method == 'CEDAS_withscore':
            labels, confidenceScores = cedas.get_labels_and_confidenceScores(STV_map)
            # sample_label
            for tid, sample_label in labels.items():
                if sample_label == 'abnormal':
                    a_pred.append(1)
                    abnormal_map[tid] = True
                    abnormal_count += 1
                else:
                    abnormal_map[tid] = False
                    a_pred.append(0)

        print('Manual labeling ratio is %.3f' % (manual_count/len(dataset)))
        print('--------------------------------')
        a_acc = accuracy_score(a_true, a_pred)
        a_recall = recall_score(a_true, a_pred)
        a_prec = precision_score(a_true, a_pred)
        a_F1_score = (2 * a_prec * a_recall)/(a_prec + a_recall)
        print('AD accuracy score is %.5f' % a_acc)
        print('AD recall score is %.5f' % a_recall)
        print('AD precision score is %.5f' % a_prec)
        print('AD F1 score is %.5f' % a_F1_score)
        print('--------------------------------')

        if abnormal_count > 8:
            print('********* RCA start *********')
            r_true.append(True)
            # 在这里对 trace 进行尾采样，若一个微簇/宏观簇的样本数越多，则采样概率低，否则采样概率高
            # 仅保留采样到的 trace id 即可

            if AD_method in ['DenStream_withscore', 'CEDAS_withscore']:
                top_list = rca(start=start, end=end, tid_list=tid_list, trace_labels=abnormal_map, confidenceScores=confidenceScores)
            else:
                top_list = rca(start=start, end=end, tid_list=tid_list, trace_labels=abnormal_map)
            
            # top_list is not empty
            if len(top_list) != 0:   
                topK = top_list[:K if len(top_list) > K else len(top_list)]
                print(f'top-{K} root cause is', topK)
                start_hour = time.localtime(start//1000).tm_hour
                chaos_service = chaos_dict.get(start_hour)

                # zhoutong add
                in_topK = True
                candidate_list = []
                for topS in topK:
                    candidate_list += topS.split('/')
                if isinstance(chaos_service, list):
                    for service in chaos_service:
                        gt_service = service.replace('-', '')[2:]
                        if gt_service not in candidate_list:
                            in_topK = False
                            break
                else:
                    gt_service = chaos_service.replace('-', '')[2:]
                    if gt_service not in candidate_list:
                        in_topK = False
                        
            # top_list is empty
            elif len(top_list) == 0:
                in_topK = False
            r_pred.append(in_topK)

        start = end
        end = start + window_duration
        # main loop end
    print('main loop end')
    
    # ========================================
    # Evaluation for AD
    # ========================================
    print('--------------------------------')
    a_acc = accuracy_score(a_true, a_pred)
    a_recall = recall_score(a_true, a_pred)
    a_prec = precision_score(a_true, a_pred)
    a_F1_score = (2 * a_prec * a_recall)/(a_prec + a_recall)
    print('AD accuracy score is %.5f' % a_acc)
    print('AD recall score is %.5f' % a_recall)
    print('AD precision score is %.5f' % a_prec)
    print('AD F1 score is %.5f' % a_F1_score)
    print('--------------------------------')

    # ========================================
    # Evaluation for RCA
    # ========================================
    print('--------------------------------')
    r_acc = accuracy_score(r_true, r_pred)
    r_recall = recall_score(r_true, r_pred)
    r_prec = precision_score(r_true, r_pred)
    print('RCA accuracy score is %.5f' % r_acc)
    print('RCA recall score is %.5f' % r_recall)
    print('RCA precision score is %.5f' % r_prec)
    print('--------------------------------')

    print("Done !")

if __name__ == '__main__':
    main()