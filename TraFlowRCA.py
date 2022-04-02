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

from DataPreprocess.STVProcess import embedding_to_vector, load_dataset, process_one_trace
from DenStream.DenStream import DenStream
from MicroRank.preprocess_data import get_span, get_service_operation_list, get_operation_slo
from MicroRank.online_rca import rca

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_str = '2022-02-27 01:00:00'
end_str = '2022-02-27 01:30:00'

def timestamp(datetime: str) -> int:
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(timeArray)) * 1000
    return ts

def main():
    # ========================================
    # Init path vector encoder
    # ========================================
    all_path = []

    # ========================================
    # Create cluster object
    # ========================================
    denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11)

    print('Start !')
    # main loop start
    while True:
        start, end = timestamp(start_str), timestamp(end_str)
        abnormal_count = 0
        abnormal_map = {}
        tid_list = []
        for tid, data in tqdm(enumerate(load_dataset(start, end)), desc="All Samples: "):
            # ========================================
            # Path vector encoder
            # ========================================
            all_path = process_one_trace(data, all_path)
            STVector = embedding_to_vector(data, all_path)

            sample_label = denstream.Cluster_AnomalyDetector(np.array(STVector), data)
            tid_list.append(tid)
            if sample_label == 'abnormal':
                abnormal_map[tid] = True
                abnormal_count += 1
        
        if abnormal_count > 8:
            rca(start, end, tid_list, abnormal_map)

        break # main loop end

    print("Done !")

if __name__ == '__main__':
    main()