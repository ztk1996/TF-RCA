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

Two_error = False
K = [1, 3, 5, 7, 10] if Two_error==False else [2, 3, 5, 7, 10]
operation_service_dict = dict()
# format stage
# start_str = '2022-04-18 21:08:00'    # changes    # '2022-01-13 00:00:00' ---> '2022-04-17 02:56:08'   '2022-04-18 21:00:00'
# start_str = '2022-04-22 22:00:00'    # changes new
# start_str = '2022-04-18 11:00:00'    # 1 abnormal
# start_str = '2022-04-19 10:42:59'    # 2 abnormal
# start_str = '2022-04-24 19:00:00'    # 2 abnormal new
# start_str = '2022-04-26 21:00:00'    # 1 abnormal new
# start_str = '2022-04-27 15:50:00'    # 1 abnormal avail
# start_str = '2022-05-01 00:00:00'    # 1 change
start_str = '2022-05-05 19:00:00'    # 1 abnormal new 5-6

window_duration = 6 * 60 * 1000 # ms
# init stage
# init_start_str = '2022-04-20 00:00:05'    # normal old
# init_end_str = '2022-04-20 09:59:55'
init_start_str = '2022-05-03 20:36:46'    # normal new

def timestamp(datetime: str) -> int:
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(timeArray)) * 1000
    return ts

def ms2str(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ms/1000))

def intersect_or_not(start1: int, end1: int, start2: int, end2: int):
    if max(start1, start2) < min(end1, end2):
        return True
    else:
        return False

def get_operation_service_pairs(dataset):
    global operation_service_dict
    for trace_id, trace in dataset.items():
        for vertex_id, vertex in trace['vertexs'].items():
            if vertex_id == '0':
                continue
            if vertex[1] not in operation_service_dict.keys():    # operation
                operation_service_dict[vertex[1]] = vertex[0]

def main():
    # ========================================
    # Init time window
    # ========================================
    init_start = timestamp(init_start_str)
    # init_end = timestamp(init_end_str)
    init_end = init_start +  13 * 60 * 60 * 1000
    
    start = timestamp(start_str)
    end = start + window_duration

    

    # ========================================
    # Init operation list
    # ========================================
    span_list = get_span(start=init_start, end=init_end, stage='init')
    # print(span_list)
    operation_list = get_service_operation_list(span_list)
    # print(operation_list)
    slo = get_operation_slo(
        service_operation_list=operation_list, span_list=span_list)
    # print(slo)

    # ========================================
    # Init evaluation for AD
    # ========================================
    a_true, a_pred = [], []

    # ========================================
    # Init evaluation for RCA
    # ========================================   
    trigger_count = 0 
    r_true_count = len(request_period_log)
    r_pred_count_0 = 0
    r_pred_count_1 = 0
    r_pred_count_2 = 0
    r_pred_count_3 = 0
    r_pred_count_4 = 0
    TP = 0    # TP ????????????????????????????????? 
    TN = 0    # TN ?????????????????????????????????
    FP = 0    # FP ????????????????????????????????????????????????
    FN = 0    # FN ????????????????????????????????????????????????

    print('Start !')
    # main loop start
    while True:
        print('--------------------------------')
        print(f'time window: {ms2str(start)} ~ {ms2str(end)}')
        abnormal_count = 0
        abnormal_map = {}
        tid_list = []
        raw_data = preprocess_span(start=start, end=end, stage='main')

        # a_true, a_pred = [], []
        span_list = get_span(start=start, end=end)
        if len(span_list) == 0:
            if start < timestamp(start_str) + (8 * 60 * 60 * 1000):
                start = end
                end = start + window_duration
                continue
            else: 
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
                    expect_duration = -1
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

        # print('--------------------------------')
        # a_acc = accuracy_score(a_true, a_pred)
        # a_recall = recall_score(a_true, a_pred)
        # a_prec = precision_score(a_true, a_pred)
        # a_F1_score = (2 * a_prec * a_recall)/(a_prec + a_recall)
        # print('AD accuracy score is %.5f' % a_acc)
        # print('AD recall score is %.5f' % a_recall)
        # print('AD precision score is %.5f' % a_prec)
        # print('AD F1 score is %.5f' % a_F1_score)
        # print('--------------------------------')

        if abnormal_count > 8:
            print('********* RCA start *********')

            trigger_count += 1

            top_list = rca_MicroRank(start=start, end=end, tid_list=tid_list, trace_labels=abnormal_map, operation_list=operation_list, slo=slo)
            
            # top_list is not empty
            if len(top_list) != 0:   
                # topK_0
                topK_0 = top_list[:K[0] if len(top_list) > K[0] else len(top_list)]
                print(f'top-{K[0]} root cause is', topK_0)
                # topK_1
                topK_1 = top_list[:K[1] if len(top_list) > K[1] else len(top_list)]
                print(f'top-{K[1]} root cause is', topK_1)
                # topK_2
                topK_2 = top_list[:K[2] if len(top_list) > K[2] else len(top_list)]
                print(f'top-{K[2]} root cause is', topK_2)
                # topK_3
                topK_3 = top_list[:K[3] if len(top_list) > K[3] else len(top_list)]
                print(f'top-{K[3]} root cause is', topK_3)
                # topK_4
                topK_4 = top_list[:K[4] if len(top_list) > K[4] else len(top_list)]
                print(f'top-{K[4]} root cause is', topK_4)

                start_hour = time.localtime(start//1000).tm_hour
                # chaos_service = span_chaos_dict.get(start_hour)
                chaos_service_list = []
                for root_cause_item in request_period_log:
                    # A: start, end    B: root_cause_item[1], root_cause_item[2]
                    # if ((root_cause_item[1]>end and root_cause_item[1]<root_cause_item[2]) or (start<end and root_cause_item[1]>root_cause_item[2]) or (start>end and start<root_cause_item[2])):
                    # if start>=root_cause_item[1] and start<=root_cause_item[2]:
                    if intersect_or_not(start1=start, end1=end, start2=root_cause_item[1], end2=root_cause_item[2]):
                        chaos_service_list.append(root_cause_item[0][0] if len(root_cause_item[0])==1 else root_cause_item[0])
                        print(f'ground truth root cause is', str(root_cause_item[0]))
                if len(chaos_service_list) == 0:
                    FP += 1
                    print("Ground truth root cause is empty !")
                    start = end
                    end = start + window_duration
                    continue
                # elif len(chaos_service_list) > 1 and Two_error == True:
                #     new_chaos_service_list = [[chaos_service_list[0], chaos_service_list[1]]]
                #     chaos_service_list = new_chaos_service_list

                in_topK_0 = False
                in_topK_1 = False
                in_topK_2 = False
                in_topK_3 = False
                in_topK_4 = False

                candidate_list_0 = []
                candidate_list_1 = []
                candidate_list_2 = []
                candidate_list_3 = []
                candidate_list_4 = []
                for topS in topK_0:
                    candidate_list_0 += topS.split('/')
                for topS in topK_1:
                    candidate_list_1 += topS.split('/')
                for topS in topK_2:
                    candidate_list_2 += topS.split('/')
                for topS in topK_3:
                    candidate_list_3 += topS.split('/')
                for topS in topK_4:
                    candidate_list_4 += topS.split('/')
                if isinstance(chaos_service_list[0], list):    # ????????????????????????
                    for service_pair in chaos_service_list:
                        if (service_pair[0].replace('-', '')[2:] in candidate_list_0) and (service_pair[1].replace('-', '')[2:] in candidate_list_0):
                            in_topK_0 = True
                        if (service_pair[0].replace('-', '')[2:] in candidate_list_1) and (service_pair[1].replace('-', '')[2:] in candidate_list_1):
                            in_topK_1 = True
                        if (service_pair[0].replace('-', '')[2:] in candidate_list_2) and (service_pair[1].replace('-', '')[2:] in candidate_list_2):
                            in_topK_2 = True
                        if (service_pair[0].replace('-', '')[2:] in candidate_list_3) and (service_pair[1].replace('-', '')[2:] in candidate_list_3):
                            in_topK_3 = True
                        if (service_pair[0].replace('-', '')[2:] in candidate_list_4) and (service_pair[1].replace('-', '')[2:] in candidate_list_4):
                            in_topK_4 = True
                else:
                    for service in chaos_service_list:
                        if service.replace('-', '')[2:] in candidate_list_0:
                            in_topK_0 = True
                        if service.replace('-', '')[2:] in candidate_list_1:
                            in_topK_1 = True
                        if service.replace('-', '')[2:] in candidate_list_2:
                            in_topK_2 = True
                        if service.replace('-', '')[2:] in candidate_list_3:
                            in_topK_3 = True
                        if service.replace('-', '')[2:] in candidate_list_4:
                            in_topK_4 = True
                
                if in_topK_0 == True:
                    r_pred_count_0 += 1
                if in_topK_1 == True:
                    r_pred_count_1 += 1
                if in_topK_2 == True:
                    r_pred_count_2 += 1
                if in_topK_3 == True:
                    r_pred_count_3 += 1
                if in_topK_4 == True:
                    r_pred_count_4 += 1

                # zhoutong add
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
        else:
            TN += 1
            for root_cause_item in request_period_log:
                if intersect_or_not(start1=start, end1=end, start2=root_cause_item[1], end2=root_cause_item[2]):
                    TN -= 1
                    FN += 1
                    break

        start = end
        end = start + window_duration
        # main loop end
    print('main loop end')
    
    print('--------------------------------')
    print("Evaluation for MicroRank")
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
    print("Top@{}:".format(K[0]))
    TP = r_pred_count_0
    if TP != 0:
        hit_rate = r_pred_count_0 / r_true_count
        # hit_rate = r_pred_count_0 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate is %.5f' % hit_rate)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('* * * * * * * *')
    print("Top@{}:".format(K[1]))
    TP = r_pred_count_1
    if TP != 0:
        hit_rate = r_pred_count_1 / r_true_count
        # hit_rate = r_pred_count_1 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate is %.5f' % hit_rate)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('* * * * * * * *')
    print("Top@{}:".format(K[2]))
    TP = r_pred_count_2
    if TP != 0:
        hit_rate = r_pred_count_2 / r_true_count
        # hit_rate = r_pred_count_2 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate is %.5f' % hit_rate)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('* * * * * * * *')
    print("Top@{}:".format(K[3]))
    TP = r_pred_count_3
    if TP != 0:
        hit_rate = r_pred_count_3 / r_true_count
        # hit_rate = r_pred_count_3 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate is %.5f' % hit_rate)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('* * * * * * * *')
    print("Top@{}:".format(K[4]))
    TP = r_pred_count_4
    if TP != 0:
        hit_rate = r_pred_count_4 / r_true_count
        # hit_rate = r_pred_count_4 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate is %.5f' % hit_rate)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('--------------------------------')

    print("Done !")

if __name__ == '__main__':
    main()