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

from DataPreprocess.STVProcess import embedding_to_vector, load_dataset, process_one_trace
from DenStream.DenStream import DenStream
from MicroRank.preprocess_data import get_span, get_service_operation_list, get_operation_slo
from MicroRank.online_rca import rca

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    print('Start !')
    # main loop start
    while True:
        print(f'time window: {ms2str(end)} ~ {ms2str(end)}')
        abnormal_count = 0
        abnormal_map = {}
        tid_list = []
        dataset = load_dataset(start, end)
        if len(dataset) == 0:
            break
        for _, data in tqdm(enumerate(dataset), desc="All Samples: "):
            # ========================================
            # Path vector encoder
            # ========================================
            all_path = process_one_trace(data, all_path)
            STVector = embedding_to_vector(data, all_path)

            sample_label = denstream.Cluster_AnomalyDetector(np.array(STVector), data)
            tid = data['trace_id']
            tid_list.append(tid)
            if sample_label == 'abnormal':
                abnormal_map[tid] = True
                abnormal_count += 1
        
        print(f'abnormal count: {abnormal_count}')
        if abnormal_count > 8:
            rca(start, end, tid_list, abnormal_map)

        start = end
        end = start + window_duration
        print()
        # main loop end

    print("Done !")

if __name__ == '__main__':
    main()