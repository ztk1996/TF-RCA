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
from MicroRank.preprocess_data import get_span, get_service_operation_list, get_operation_slo, get_operation_duration_data
from MicroRank.online_rca import rca_MicroRank
from DataPreprocess.params import span_chaos_dict, request_period_log
from DataPreprocess.SpanProcess import preprocess_span
import warnings
warnings.filterwarnings("ignore")
import sys
MAX_INT = sys.maxsize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K = 3
start_str = '2022-04-16 20:08:00'    # '2022-01-13 00:00:00' ---> '2022-04-17 02:56:08'
window_duration = 20 * 60 * 1000 # ms
init_window_duration = 5 * 60 * 1000 # ms

def timestamp(datetime: str) -> int:
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(timeArray)) * 1000
    return ts

def ms2str(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ms/1000))

def main():
    # ========================================
    # Init time window
    # ========================================
    start = timestamp(start_str)
    end = start + init_window_duration

    # ========================================
    # Init operation list
    # ========================================
    span_list = get_span(start=start, end=end)
    # print(span_list)
    operation_list = get_service_operation_list(span_list)
    # print(operation_list)
    slo = get_operation_slo(
        service_operation_list=operation_list, span_list=span_list)
    # print(slo)
    start = end
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
        tid_list = []
        raw_data = preprocess_span(start=start, end=end, stage='main')

        a_true, a_pred = [], []
        span_list = get_span(start=start, end=end)
        if len(span_list) == 0:
            print("Error: Current span list is empty ")
            break
        #operation_list = get_service_operation_list(span_list)
        operation_count = get_operation_duration_data(operation_list, span_list)

        for trace_id in operation_count:
            if trace_id not in raw_data.keys():
                continue
            else:    
                a_true.append(raw_data[trace_id]['abnormal'])
                tid_list.append(trace_id)
                
            real_duration = float(operation_count[trace_id]['duration']) / 1000.0
            expect_duration = 0.0
            for operation in operation_count[trace_id]:
                if "duration" == operation:
                    continue
                if operation not in slo:
                    expect_duration = MAX_INT
                    break
                else:
                    expect_duration += operation_count[trace_id][operation] * (
                        slo[operation][0] + 1.5 * slo[operation][1])

            # if real_duration > expect_duration:
            if raw_data[trace_id]['abnormal']:
                a_pred.append(1)
                abnormal_map[trace_id] = True
                abnormal_count += 1
            else:
                abnormal_map[trace_id] = False
                a_pred.append(0)

        print("anormaly_count:", abnormal_count)

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

        if abnormal_count > 4:
            print('********* RCA start *********')
            r_true.append(True)

            top_list = rca_MicroRank(start=start, end=end, tid_list=tid_list, trace_labels=abnormal_map, operation_list=operation_list, slo=slo)
            
            # top_list is not empty
            if len(top_list) != 0:   
                topK = top_list[:K if len(top_list) > K else len(top_list)]
                print(f'top-{K} root cause is', topK)
                start_hour = time.localtime(start//1000).tm_hour
                # chaos_service = span_chaos_dict.get(start_hour)
                chaos_service_list = []
                for root_cause_item in request_period_log:
                    # A: start, end    B: root_cause_item[1], root_cause_item[2]
                    if ((root_cause_item[1]>end and root_cause_item[1]<root_cause_item[2]) or (start<end and root_cause_item[1]>root_cause_item[2]) or (start>end and start<root_cause_item[2])):
                    # if start>=root_cause_item[1] and start<=root_cause_item[2]:
                        chaos_service_list.append(root_cause_item[0][0])
                        print(f'ground truth root cause is', root_cause_item[0][0])


                # # zhoutong add
                # in_topK = True
                # candidate_list = []
                # for topS in topK:
                #     candidate_list += topS.split('/')
                # if isinstance(chaos_service, list):
                #     for service in chaos_service:
                #         gt_service = service.replace('-', '')[2:]
                #         if gt_service not in candidate_list:
                #             in_topK = False
                #             break
                # else:
                #     gt_service = chaos_service.replace('-', '')[2:]
                #     if gt_service not in candidate_list:
                #         in_topK = False
                        
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