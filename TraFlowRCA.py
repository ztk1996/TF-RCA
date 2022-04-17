from importlib.resources import path
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
import sys
import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score
from DataPreprocess.STVProcess import embedding_to_vector, load_dataset, process_one_trace
from DenStream.DenStream import DenStream
from CEDAS.CEDAS import CEDAS
from MicroRank.preprocess_data import get_span, get_service_operation_list, get_operation_slo
from MicroRank.online_rca import rca
from DataPreprocess.params import span_chaos_dict, trace_chaos_dict
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K = 3
all_path = dict()
manual_labels_list = []    # 人工标注为正常的 trace id 列表 manual_labels_list : [trace_id1, trace_id2, ...]
first_tag = True
start_str = '2022-01-13 00:00:00'    # trace: '2022-02-25 00:00:00', span: '2022-01-13 00:00:00'
init_start_str = '2022-01-13 00:00:00'    # normal traces  trace: '2022-02-25 00:00:00', span: '2022-01-13 00:00:00'
window_duration = 60 * 60 * 1000    # ms
AD_method = 'DenStream_withscore'    # 'DenStream_withscore', 'DenStream_withoutscore', 'CEDAS_withscore', 'CEDAS_withoutscore'
Sample_method = 'none'    # 'none', 'micro', 'macro'
dataLevel = 'span'    # 'trace', 'span'
path_decay = 0.01
path_thres = 0.5
reCluster_thres = 0.1

def timestamp(datetime: str) -> int:
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(timeArray)) * 1000
    return ts

def ms2str(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ms/1000))

def simplify_cluster(cluster_obj, dataset, cluster_status, data_status):    # dataset: [[STVector1, sample_info1], [STVector2, sample_info2]]
    global all_path
    global first_tag
    manual_count = 0
    for data in tqdm(dataset, desc="Cluster Samples: "):
        # ========================================
        # Path vector encoder
        # ========================================
        if cluster_status == 'init':
            for status in all_path.values():
                status[0] -= path_decay
            all_path = process_one_trace(data, all_path)
            STVector = embedding_to_vector(data, all_path)
        elif cluster_status == 'reCluster':
            STVector = data[0]

        if AD_method in ['DenStream_withoutscore', 'DenStream_withscore']:
            sample_label, label_status = cluster_obj.Cluster_AnomalyDetector(sample=np.array(STVector), sample_info=data if cluster_status=='init' else data[1], data_status=data_status, manual_labels_list=manual_labels_list)
        elif AD_method in ['CEDAS_withoutscore', 'CEDAS_withscore']:
            if first_tag:
                # 1. Initialization
                sample_label, label_status = cluster_obj.initialization(sample=np.array(STVector), sample_info=data if cluster_status=='init' else data[1], data_status=data_status, manual_labels_list=manual_labels_list)
                first_tag = False
            else:
                cluster_obj.changed_cluster = None
                # 2. Update Micro-Clusters
                sample_label, label_status = cluster_obj.Cluster_AnomalyDetector(sample=np.array(STVector), sample_info=data if cluster_status=='init' else data[1], data_status=data_status, manual_labels_list=manual_labels_list)
                # 3. Kill Clusters
                cluster_obj.kill()
                if cluster_obj.changed_cluster and cluster_obj.changed_cluster.count > cluster_obj.threshold:
                    # 4. Update Cluster Graph
                    cluster_obj.update_graph()

        # label_status
        if label_status == 'manual':
            manual_count += 1


def init_Cluster(cluster_obj, init_start_str):
    # ========================================
    # Init time window
    # ========================================
    start = timestamp(init_start_str)     # + 1 * 60 * 1000
    end = start + window_duration
    timeWindow_count = 0

    # ========================================
    # Init Data loader
    # ========================================
    if dataLevel == 'trace':
        print("Init Data loading ...")
        file = open(r'/data/TraceCluster/RCA/total_data/test.json', 'r')
        raw_data_total = json.load(file)
        print("Finish init data load !")

    print('Init Start !')
    # Init main loop start
    while True:
        print('--------------------------------')
        print(f'time window: {ms2str(start)} ~ {ms2str(end)}')
        # temp
        if dataLevel == 'trace':
            if str(ms2str(start)) not in ['2022-02-25 01:00:00', '2022-02-25 02:00:00', '2022-02-25 03:00:00', '2022-02-25 05:00:00', '2022-02-25 09:00:00', '2022-02-25 11:00:00', '2022-02-25 13:00:00', '2022-02-25 14:00:00', '2022-02-25 17:00:00', '2022-02-25 18:00:00', '2022-02-25 19:00:00']:
                start = end
                end = start + window_duration
                continue
        timeWindow_count += 1

        if dataLevel == 'span':
            dataset, raw_data_dict = load_dataset(start, end, dataLevel, 'init')
        elif dataLevel == 'trace':
            dataset, raw_data_dict = load_dataset(start, end, dataLevel, 'init', raw_data_total)
            
        if len(dataset) == 0:
            break
        
        # do cluster
        simplify_cluster(cluster_obj=cluster_obj, dataset=dataset, cluster_status='init', data_status='init')

        start = end
        end = start + window_duration

        delete_index_candidate = [status[1] for status in all_path.values() if status[0]<path_thres]
        if len(delete_index_candidate) / len(all_path) >= reCluster_thres:
            do_reCluster(cluster_obj=cluster_obj, data_status='init')
    print('Init finish !')

        

def do_reCluster(cluster_obj, data_status, label_map_reCluster=dict()):
    print("reCluster Start ...")
    global all_path
    global first_tag
    delete_index = [status[1] for status in all_path.values() if status[0]<path_thres]
    
    # adjust all STVectors
    reCluster_dataset = list()    # [[STVector1, sample_info1], [STVector2, sample_info2]]
    for cluster in cluster_obj.p_micro_clusters+cluster_obj.o_micro_clusters if AD_method in ['DenStream_withoutscore', 'DenStream_withscore'] else cluster_obj.micro_clusters:
        for data_item in cluster.members.values():    # data_item: [STVector, sample_info]
            new_STVector = []
            for idx, value in enumerate(data_item[0]):
                if idx not in delete_index:
                    new_STVector.append(value)
            # 若一个 STVector 被删成空或者全0，则丢弃这个 STVector
            if len(new_STVector)!=0 and new_STVector.count(0)!=len(new_STVector):
                reCluster_dataset.append([np.array(new_STVector), data_item[1]])
    reCluster_dataset.sort(key=lambda i: i[1]['time_stamp'])
    print("reCluster dataset length: ", len(reCluster_dataset))
    
    # adjust all_path dict
    new_all_path = dict()
    for path, path_status in sorted(all_path.items(), key = lambda item: item[1][1]):
        if path_status[1] not in delete_index:
            new_all_path[path] = [path_status[0], len(new_all_path)]
    all_path = new_all_path

    # clear all clusters
    if AD_method in ['DenStream_withoutscore', 'DenStream_withscore']:
        cluster_obj.p_micro_clusters.clear()
        cluster_obj.o_micro_clusters.clear()
    elif AD_method in ['CEDAS_withoutscore', 'CEDAS_withscore']:
        first_tag = True
        cluster_obj.micro_clusters.clear()

    # do recluster
    simplify_cluster(cluster_obj=cluster_obj, dataset=reCluster_dataset, cluster_status='reCluster', data_status=data_status)
    print("reCluster Finish !")



def main():
    # ========================================
    # Init path vector encoder
    # all_path = {path1: [energy1, index1], path2: [energy2, index2]}
    # label_map_reCluster = {trace_id1: label1, trace_id2: label2}
    # ========================================
    global all_path
    global first_tag
    global manual_labels_list
    label_map_reCluster = dict()

    # ========================================
    # Create cluster object
    # Init clusters
    # ========================================
    if AD_method in ['DenStream_withscore', 'DenStream_withoutscore']:
        # denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11)
        denstream = DenStream(eps=100, lambd=0.1, beta=0.2, mu=6)
        init_Cluster(denstream, init_start_str)
    elif AD_method in ['CEDAS_withscore', 'CEDAS_withoutscore']:
        cedas = CEDAS(r0=100, decay=0.001, threshold=5)
        first_tag = True
        init_Cluster(cedas, init_start_str)

    # ========================================
    # Init time window
    # ========================================
    start = timestamp(start_str)     # + 1 * 60 * 1000
    end = start + window_duration
    timeWindow_count = 0

    # ========================================
    # Init evaluation for AD
    # ========================================
    a_true, a_pred = [], []

    # ========================================
    # Init evaluation for RCA
    # ========================================
    r_true, r_pred = [], []    

    # ========================================
    # Init root cause pattern
    # ========================================
    rc_pattern = []

    # ========================================
    # Data loader
    # ========================================
    if dataLevel == 'trace':
        print("Main Data loading ...")
        file = open(r'/data/TraceCluster/RCA/total_data/test.json', 'r')
        raw_data_total = json.load(file)
        print("Finish main data load !")

    print('Start !')
    # main loop start
    while True:
        print('--------------------------------')
        print(f'time window: {ms2str(start)} ~ {ms2str(end)}')
        # temp
        if dataLevel == 'trace':
            if str(ms2str(start)) not in ['2022-02-25 01:00:00', '2022-02-25 02:00:00', '2022-02-25 03:00:00', '2022-02-25 05:00:00', '2022-02-25 09:00:00', '2022-02-25 11:00:00', '2022-02-25 13:00:00', '2022-02-25 14:00:00', '2022-02-25 17:00:00', '2022-02-25 18:00:00', '2022-02-25 19:00:00']:
                start = end
                end = start + window_duration
                continue
        timeWindow_count += 1
        abnormal_count = 0
        abnormal_map = {}
        STV_map_window = {}
        tid_list = []
        if dataLevel == 'span':
            dataset, raw_data_dict = load_dataset(start, end, dataLevel, 'main')
        elif dataLevel == 'trace':
            dataset, raw_data_dict = load_dataset(start, end, dataLevel, 'main', raw_data_total)
            
        if len(dataset) == 0:
            break
        
        a_true, a_pred = [], []
        # Init manual count
        manual_count = 0
        for _, data in tqdm(enumerate(dataset), desc="Time Window Samples: "):
            # ========================================
            # Path vector encoder
            # ========================================
            for status in all_path.values():
                status[0] -= path_decay
            all_path = process_one_trace(data, all_path)
            STVector = embedding_to_vector(data, all_path)
            STV_map_window[data['trace_id']] = np.array(STVector)

            a_true.append(data['trace_bool'])

            if AD_method in ['DenStream_withoutscore', 'DenStream_withscore']:
                sample_label, label_status = denstream.Cluster_AnomalyDetector(sample=np.array(STVector), sample_info=data, data_status='main', manual_labels_list=manual_labels_list)
                if label_status == 'manual':
                    label_map_reCluster[data['trace_id']] = sample_label
            elif AD_method in ['CEDAS_withoutscore', 'CEDAS_withscore']:
                if first_tag:
                    # 1. Initialization
                    sample_label, label_status = cedas.initialization(sample=np.array(STVector), sample_info=data, data_status='main', manual_labels_list=manual_labels_list)
                    if label_status == 'manual':
                        label_map_reCluster[data['trace_id']] = sample_label
                    first_tag = False
                else:
                    cedas.changed_cluster = None
                    # 2. Update Micro-Clusters
                    sample_label, label_status = cedas.Cluster_AnomalyDetector(sample=np.array(STVector), sample_info=data, data_status='main', manual_labels_list=manual_labels_list)
                    if label_status == 'manual':
                        label_map_reCluster[data['trace_id']] = sample_label
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
            manual_labels_list = denstream.update_cluster_labels(manual_labels_list)
            labels, confidenceScores, sampleRates = denstream.get_labels_confidenceScores_sampleRates(STV_map=STV_map_window, cluster_type=Sample_method)
            # sample_label
            for tid, sample_label in labels.items():
                if sample_label == 'abnormal':
                    a_pred.append(1)
                    abnormal_map[tid] = True
                    abnormal_count += 1
                else:
                    abnormal_map[tid] = False
                    a_pred.append(0)
            AD_pattern = [micro_cluster for micro_cluster in denstream.p_micro_clusters+denstream.o_micro_clusters if micro_cluster.AD_selected==True]
        elif AD_method == 'CEDAS_withscore':
            manual_labels_list = cedas.update_cluster_labels(manual_labels_list)
            labels, confidenceScores, sampleRates = cedas.get_labels_confidenceScores_sampleRates(STV_map=STV_map_window, cluster_type=Sample_method)
            # sample_label
            for tid, sample_label in labels.items():
                if sample_label == 'abnormal':
                    a_pred.append(1)
                    abnormal_map[tid] = True
                    abnormal_count += 1
                else:
                    abnormal_map[tid] = False
                    a_pred.append(0)
            AD_pattern = [micro_cluster for micro_cluster in cedas.micro_clusters if micro_cluster.AD_selected==True]


        print('Manual labeling count is ', manual_count)
        print('Manual labeling ratio is %.3f' % (manual_count/len(dataset)))
        print('--------------------------------')
        a_acc = accuracy_score(a_true, a_pred)
        a_prec = precision_score(a_true, a_pred)
        a_recall = recall_score(a_true, a_pred)
        a_F1_score = (2 * a_prec * a_recall)/(a_prec + a_recall)
        print('AD accuracy score is %.5f' % a_acc)
        print('AD precision score is %.5f' % a_prec)
        print('AD recall score is %.5f' % a_recall)
        print('AD F1 score is %.5f' % a_F1_score)
        print('--------------------------------')

        pattern_IoU = len(set(AD_pattern)&set(rc_pattern)) / len(set(AD_pattern)|set(rc_pattern)) if len(set(AD_pattern)|set(rc_pattern)) != 0 else -1

        if abnormal_count > 8 and pattern_IoU < 0.5:
            rc_pattern = AD_pattern

            print('********* RCA start *********')
            r_true.append(True)

            if AD_method in ['DenStream_withscore', 'CEDAS_withscore']:
                # 在这里对 trace 进行尾采样，若一个微簇/宏观簇的样本数越多，则采样概率低，否则采样概率高
                # 仅保留采样到的 trace id 即可
                sampled_tid_list = []
                for tid in tid_list:
                    if sampleRates[tid] >= np.random.uniform(0, 1):
                        sampled_tid_list.append(tid)
                if len(sampled_tid_list) != 0:
                    top_list = rca(start=start, end=end, tid_list=sampled_tid_list, trace_labels=abnormal_map, traces_dict=raw_data_dict, confidenceScores=confidenceScores, dataLevel=dataLevel)
                else:
                    top_list = []
            else:
                top_list = rca(start=start, end=end, tid_list=tid_list, trace_labels=abnormal_map, traces_dict=raw_data_dict, dataLevel=dataLevel)
            
            # top_list is not empty
            if len(top_list) != 0:   
                topK = top_list[:K if len(top_list) > K else len(top_list)]
                print(f'top-{K} root cause is', topK)
                start_hour = time.localtime(start//1000).tm_hour
                if dataLevel == 'span':
                    chaos_service = span_chaos_dict.get(start_hour)
                elif dataLevel == 'trace':
                    chaos_service = trace_chaos_dict.get(start_hour)
                print(f'ground truth root cause is', chaos_service)

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
        else:
            rc_pattern = []

        start = end
        end = start + window_duration

        # reCluster ...
        delete_index_candidate = [status[1] for status in all_path.values() if status[0]<path_thres]
        if len(delete_index_candidate) / len(all_path) >= reCluster_thres:
            if AD_method in ['DenStream_withoutscore', 'DenStream_withscore']:
                # if data_status is 'main', all new cluster labels are 'abnormal' and label_status is 'auto'
                # if data_status is 'init', all new cluster labels are 'normal' and label_status is 'auto'
                do_reCluster(cluster_obj=denstream, data_status='main', label_map_reCluster=label_map_reCluster)
            else:
                do_reCluster(cluster_obj=cedas, data_status='main', label_map_reCluster=label_map_reCluster)
        
        # visualization ...
        if AD_method in ['DenStream_withoutscore', 'DenStream_withscore']:
            if len((denstream.p_micro_clusters+denstream.o_micro_clusters)) > 1:
                denstream.visualization_tool()
        else:
            if len(cedas.micro_clusters) > 1:
                cedas.visualization_tool()
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