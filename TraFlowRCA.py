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
from MicroRank.preprocess_data import get_span, get_service_operation_list, get_operation_slo
from MicroRank.online_rca import rca
from DataPreprocess.params import chaos_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K = 3
start_str = '2022-03-01 11:00:00'
window_duration = 5 * 60 * 1000 # ms

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
    denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11)
    start = timestamp(start_str)
    end = start + window_duration

    r_true, r_pred = [], []    

    print('Start !')
    # main loop start
    while True:
        print('--------------------------------')
        print(f'time window: {ms2str(end)} ~ {ms2str(end)}')
        abnormal_count = 0
        abnormal_map = {}
        tid_list = []
        dataset = load_dataset(start, end)
        if len(dataset) == 0:
            break

        a_true, a_pred = [], []
        for _, data in tqdm(enumerate(dataset), desc="All Samples: "):
            # ========================================
            # Path vector encoder
            # ========================================
            all_path = process_one_trace(data, all_path)
            STVector = embedding_to_vector(data, all_path)

            a_true.append(data['trace_bool'])
            sample_label = denstream.Cluster_AnomalyDetector(np.array(STVector), data)
            tid = data['trace_id']
            tid_list.append(tid)
            if sample_label == 'abnormal':
                a_pred.append(1)
                abnormal_map[tid] = True
                abnormal_count += 1
            else:
                a_pred.append(0)
        
        acc = accuracy_score(a_true, a_pred)
        recall = recall_score(a_true, a_pred)
        prec = precision_score(a_true, a_pred)
        print(f'abnormal count: {abnormal_count}')
        print('accuracy score is %.5f' % acc)
        print('recall score is %.5f' % recall)
        print('precision score is %.5f' % prec)

        if abnormal_count > 8:
            r_true.append(True)
            top_list = rca(start, end, tid_list, abnormal_map)
            topK = top_list[:K if len(top_list) > K else len(top_list)]
            print('top-K is', topK)
            start_hour = time.localtime(start//1000).tm_hour
            chaos_service = chaos_dict.get(start_hour)

            for i in range(0, len(topK)):
                # TODO
                pass

            in_topK = True
            if isinstance(chaos_service, list):
                for c in chaos_service:
                    if c not in topK:
                        in_topK = False
                        break
            elif chaos_service not in topK:
                in_topK = False
            
            r_pred.append(in_topK)



        start = end
        end = start + window_duration
        # main loop end

    print('main loop end')
    acc = accuracy_score(r_true, r_pred)
    recall = recall_score(r_true, r_pred)
    prec = precision_score(r_true, r_pred)
    print('accuracy score is %.5f' % acc)
    print('recall score is %.5f' % recall)
    print('precision score is %.5f' % prec)
    print("Done !")

if __name__ == '__main__':
    main()