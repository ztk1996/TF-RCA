from importlib.resources import path
from multiprocessing import connection
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
from DataPreprocess.STVProcess import embedding_to_vector, load_dataset, process_one_trace, check_match, get_service_slo
from DenStream.DenStream import DenStream
from CEDAS.CEDAS import CEDAS
from MicroRank.preprocess_data import get_span, get_service_operation_list, get_operation_slo
from MicroRank.online_rca import rca
from DataPreprocess.params import span_chaos_dict, trace_chaos_dict, request_period_log, normal_change_log
from db_utils import *
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Two_error = False
K = [1, 3, 5, 7, 10] if Two_error==False else [2, 3, 5, 7, 10]
all_path = dict()
operation_service_dict = dict()
RCA_level = 'service'    # operation
use_manual = False
manual_labels_list = []
# manual_labels_list = ['617d5c02352849119c2e5df8b70fa007.36.16503361793140001', 'dda6af5c49d546068432825e17b981aa.38.16503361824380001', '27a94f3afad745c69963997831868eb1.38.16503362398950001', '15480e1347c147a086b68221ca743874.38.16503369859250001', '262b9727d1584947a02905150a089faa.38.16503382599320123', 'ab212da6fff042febb91b313658a0005.46.16503384128150203', '0b225e568e304836a7901e0cff56205a.39.16503393835170053', '262b9727d1584947a02905150a089faa.39.16503397746270231']    # 人工标注为正常的 trace id 列表 manual_labels_list : [trace_id1, trace_id2, ...]
first_tag = True
# start_str = '2022-04-19 10:42:59'    # '2022-04-18 21:08:00' # '2022-04-19 10:42:59'    # trace: '2022-02-25 00:00:00', '2022-04-16 20:08:03', '2022-04-18 11:00:00', '2022-04-18 21:00:00'; span: '2022-01-13 00:00:00'
# format stage
# start_str = '2022-04-18 21:08:00'    # changes
# start_str = '2022-04-22 22:00:00'    # changes new
# start_str = '2022-04-18 11:00:00'    # 1 abnormal
# start_str = '2022-04-19 10:42:59'    # 2 abnormal
# start_str = '2022-04-24 19:00:00'    # 2 abnormal new
# start_str = '2022-04-26 21:00:00'    # 1 abnormal new 2022-04-26 21:02:22
# start_str = '2022-04-28 12:00:00'
# start_str = '2022-04-27 15:50:00'    # 1 abnormal avail
# start_str = '2022-05-01 00:00:00'    # 1 change
# start_str = '2022-04-28 12:00:00'    # 2 abnormal
# start_str = '2022-05-05 19:00:00'    # 1 abnormal new 5-6
start_str = '2022-05-09 15:00:00'    # change new 5-10

# init stage
# init_start_str = '2022-04-18 00:00:05'    # normal old
init_start_str = '2022-04-26 08:22:09'    # normal new
window_duration = 6 * 60 * 1000    # ms
window_duration_init = 6 * 60 * 1000 / 4    # ms
check_window = 5 * 60 * 1000    # ms
AD_method = 'DenStream_withoutscore'    # 'DenStream_withscore', 'DenStream_withoutscore', 'CEDAS_withscore', 'CEDAS_withoutscore'
Sample_method = 'none'    # 'none', 'micro', 'macro', 'rate'
dataLevel = 'trace'    # 'trace', 'span'
path_decay = 0.01
path_thres = 0.0
reCluster_thres = 0.1

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
            # if use_manual == True and cluster_status == 'init':
            #     cluster_obj.update_cluster_and_trace_tables()
            sample_label, label_status = cluster_obj.Cluster_AnomalyDetector(sample=np.array(STVector), sample_info=data if cluster_status=='init' else data[1], data_status=data_status, manual_labels_list=manual_labels_list, stage=cluster_status)
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
    end = start + window_duration_init
    timeWindow_count = 0

    # ========================================
    # Init Data loader
    # ========================================
    if dataLevel == 'trace':
        print("Init Data loading ...")
        # file = open(r'/data/TraceCluster/RCA/total_data/test.json', 'r')
        # file = open(r'/home/kagaya/work/TF-RCA/DataPreprocess/data/preprocessed/trainticket/2022-04-20_17-34-08/data.json', 'r')    # old
        file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-08_14-03-38/data.json', 'r')    # new
        raw_data_total = json.load(file)
        get_operation_service_pairs(raw_data_total)
        get_service_slo(raw_data=raw_data_total)
        print("Finish init data load !")

    print('Init Start !')
    # Init main loop start
    while True:
        print('--------------------------------')
        print(f'time window: {ms2str(start)} ~ {ms2str(end)}')
        # temp
        # if dataLevel == 'trace':
        #     if str(ms2str(start)) not in ['2022-02-25 01:00:00', '2022-02-25 02:00:00', '2022-02-25 03:00:00', '2022-02-25 05:00:00', '2022-02-25 09:00:00', '2022-02-25 11:00:00', '2022-02-25 13:00:00', '2022-02-25 14:00:00', '2022-02-25 17:00:00', '2022-02-25 18:00:00', '2022-02-25 19:00:00']:
        #         start = end
        #         end = start + window_duration
        #         continue
        timeWindow_count += 1

        if dataLevel == 'span':
            dataset, raw_data_dict = load_dataset(start, end, dataLevel, 'init')
        elif dataLevel == 'trace':
            dataset, raw_data_dict = load_dataset(start, end, dataLevel, 'init', raw_data_total)

        if len(dataset) == 0:
            if start < timestamp(init_start_str) + (8 * 60 * 60 * 1000):
                start = end
                end = start + window_duration_init
                continue
            else: 
                break
        
        # do cluster
        simplify_cluster(cluster_obj=cluster_obj, dataset=dataset, cluster_status='init', data_status='init')

        start = end
        end = start + window_duration_init

        delete_index_candidate = [status[1] for status in all_path.values() if status[0]<path_thres]
        if len(delete_index_candidate) / len(all_path) >= reCluster_thres:
            do_reCluster(cluster_obj=cluster_obj, data_status='init', current_time=start)

        if timeWindow_count >= int(1 * 60 * 60 * 1000 / window_duration_init):
            break
    
    # update cluster table when finish init
    if use_manual == True:
        cluster_obj.update_cluster_and_trace_tables()
    print('Init finish !')

        

def do_reCluster(cluster_obj, data_status, current_time, label_map_reCluster=dict()):
    print("reCluster Start ...")
    global all_path
    global first_tag
    delete_index = [status[1] for status in all_path.values() if status[0]<path_thres]
    delete_index_all = list()
    for index in delete_index:
        delete_index_all.append(index*2)    # path_index
        delete_index_all.append(index*2+1)    # statusCode_index
    
    # # adjust all STVectors
    # reCluster_dataset = list()    # [[STVector1, sample_info1], [STVector2, sample_info2]]
    # for cluster in cluster_obj.p_micro_clusters+cluster_obj.o_micro_clusters if AD_method in ['DenStream_withoutscore', 'DenStream_withscore'] else cluster_obj.micro_clusters:
    #     for data_item in cluster.members.values():    # data_item: [STVector, sample_info]
    #         new_STVector = []
    #         for idx, value in enumerate(data_item[0]):
    #             if int(idx/2) not in delete_index:
    #                 new_STVector.append(value)    # rt dimension & isError dimension
    #         # 若一个 STVector 被删成空或者全0，则丢弃这个 STVector
    #         if len(new_STVector)!=0 and new_STVector.count(0)!=len(new_STVector):
    #             reCluster_dataset.append([np.array(new_STVector), data_item[1]])
    # reCluster_dataset.sort(key=lambda i: i[1]['time_stamp'])
    # print("reCluster dataset length: ", len(reCluster_dataset))

    # # adjust all STVectors
    # reCluster_dataset = list()    # [[STVector1, sample_info1], [STVector2, sample_info2]]
    # for cluster in cluster_obj.p_micro_clusters+cluster_obj.o_micro_clusters if AD_method in ['DenStream_withoutscore', 'DenStream_withscore'] else cluster_obj.micro_clusters:
    #     for data_item in cluster.members.values():    # data_item: [STVector, sample_info]
    #         # 将1小时前的数据丢弃
    #         if data_item[1]['time_stamp']<current_time-(1 * 60 * 60 * 1000):
    #             continue
    #         new_STVector = []
    #         for idx, value in enumerate(data_item[0]):
    #             if int(idx/2) not in delete_index:
    #                 new_STVector.append(value)    # rt dimension & isError dimension
    #         # 若一个 STVector 被删成空或者全0，则丢弃这个 STVector
    #         if len(new_STVector)!=0 and new_STVector.count(0)!=len(new_STVector):
    #             reCluster_dataset.append([np.array(new_STVector), data_item[1]])
    # reCluster_dataset.sort(key=lambda i: i[1]['time_stamp'])
    # print("reCluster dataset length: ", len(reCluster_dataset))
    
    # adjust all STVectors
    reCluster_dataset = list()    # [[STVector1, sample_info1], [STVector2, sample_info2]]
    for cluster in cluster_obj.p_micro_clusters+cluster_obj.o_micro_clusters if AD_method in ['DenStream_withoutscore', 'DenStream_withscore'] else cluster_obj.micro_clusters:
        for data_item in cluster.members.values():    # data_item: [STVector, sample_info]
            # 将1小时前的数据丢弃
            # if data_item[1]['time_stamp']<current_time-(1 * 60 * 60 * 1000):
            if data_item[1]['time_stamp']<current_time-(1 * 60 * 60 * 1000 if data_status=='main' else 1 * 60 * 60 * 1000 / 4):
                continue
            new_STVector = np.append(data_item[0], [0]*(len(cluster.mean)-len(data_item[0])))
            new_STVector = list(np.delete(new_STVector, delete_index_all, axis=None))
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
        # if use_manual == True:
        #     db_delete_all()
    elif AD_method in ['CEDAS_withoutscore', 'CEDAS_withscore']:
        first_tag = True
        cluster_obj.micro_clusters.clear()

    # do recluster
    simplify_cluster(cluster_obj=cluster_obj, dataset=reCluster_dataset, cluster_status='reCluster', data_status=data_status)
    
    # assign cluster label
    if len(label_map_reCluster) != 0:
        new_label_map_reCluster = dict()
        for cluster in cluster_obj.p_micro_clusters+cluster_obj.o_micro_clusters:
            normal_count = 0
            abnormal_count = 0
            for trace_id in cluster.members.keys():
                if trace_id not in label_map_reCluster.keys():
                    normal_count += 1
                elif trace_id in label_map_reCluster.keys():
                    # delete useless label items
                    new_label_map_reCluster[trace_id] = label_map_reCluster[trace_id]
                    if label_map_reCluster[trace_id] == 'normal':
                        normal_count += 1
                    else:
                        abnormal_count += 1
            if normal_count >= abnormal_count:
                cluster.label = 'normal'
                # if use_manual == True:
                #     db_update_label(cluster_id=id(cluster), cluster_label=cluster.label)
            else:
                cluster.label = 'abnormal'
                # if use_manual == True:
                #     db_update_label(cluster_id=id(cluster), cluster_label=cluster.label)
        label_map_reCluster = new_label_map_reCluster

    print("reCluster Finish !")



def main():
    # manual labels list
    traceID_list = list()

    # ========================================
    # Init path vector encoder
    # all_path = {path1: [energy1, index1], path2: [energy2, index2]}
    # label_map_reCluster = {trace_id1: label1, trace_id2: label2}
    # ========================================
    # differences_count, differences_svc_count, ab_count, ad_count_list, root_cause_check_dict, in_rate, in_count_normal_dict, case_data_num, case_pattern, label_error_traces_init, label_error_traces_format, case_service_pattern = check_match()
    global all_path
    global first_tag
    global manual_labels_list
    label_map_reCluster = dict()

    # ========================================
    # Create db table
    # ========================================
    # cluster_sql = '''
    #         create table clusters(
    #             id int AUTO_INCREMENT  primary key not null,
    #             cluster_id varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci  not null,
    #             creation_time varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci  not null,
    #             cluster_label varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci  not null,
    #             member_count int  not null,
    #             member_ids varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci  not null,
    #             cluster_weight float  not null,
    #         )
    # '''
    # cur.execute(cluster_sql)
    # db_connect.commit()

    # ========================================
    # Create cluster object
    # Init clusters
    # ========================================
    if AD_method in ['DenStream_withscore', 'DenStream_withoutscore']:
        # denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11)
        denstream = DenStream(eps=1, lambd=0.1, beta=0.2, mu=6, use_manual=use_manual)    # eps=80    beta=0.2   mu=6
        init_Cluster(denstream, init_start_str)
    elif AD_method in ['CEDAS_withscore', 'CEDAS_withoutscore']:
        cedas = CEDAS(r0=100, decay=0.001, threshold=5)
        first_tag = True
        init_Cluster(cedas, init_start_str)

    # ========================================
    # Init time window
    # ========================================
    # start_buffer <---6min---> start <---6min---> end <---6min---> end_buffer
    start = timestamp(start_str)     # + 1 * 60 * 1000
    end = start + window_duration
    start_buffer = start - window_duration
    end_buffer = end + window_duration
    timeWindow_count = 0

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
    TP = 0    # TP 是预测为正类且预测正确 
    TN = 0    # TN 是预测为负类且预测正确
    FP = 0    # FP 是把实际负类分类（预测）成了正类
    FN = 0    # FN 是把实际正类分类（预测）成了负类

    # ========================================
    # Init root cause pattern
    # ========================================
    rc_pattern = []

    # ========================================
    # Data loader
    # ========================================
    if dataLevel == 'trace':
        print("Main Data loading ...")
        # file = open(r'/data/TraceCluster/RCA/total_data/test.json', 'r')
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-17_15-29-46/data.json', 'r')
        # file = open(r'/home/kagaya/work/TF-RCA/DataPreprocess/data/preprocessed/trainticket/2022-04-19_10-05-14/data.json', 'r')    # 1 abnormal
        # file = open(r'/home/kagaya/work/TF-RCA/DataPreprocess/data/preprocessed/trainticket/2022-04-19_21-01-30/data.json', 'r')    # 2 abnormal
        # file = open(r'/home/kagaya/work/TF-RCA/DataPreprocess/data/preprocessed/trainticket/2022-04-19_11-34-58/data.json', 'r')    # change
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-23_13-34-27/data.json', 'r')    # change new
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-25_12-40-13/data.json', 'r')    # abnormal2 new
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-27_22-19-20/data.json', 'r')    # abnormal1 new new
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-27_12-58-19/data.json', 'r')    # abnormal1 new
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-28_21-01-41/data.json', 'r')
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-28_22-55-23/data.json', 'r')
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-29_21-45-19/data.json', 'r')
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-30_14-39-40/data.json', 'r')    # abnormal 1 avail
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-01_13-40-58/data.json', 'r')    # change 1
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-30_19-41-29/data.json', 'r')    # abnormal 2
        # file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-06_17-28-43/data.json', 'r')    # 1 abnormal new 5-6
        file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-11_00-06-32/data.json', 'r')    # change new 5-10
        raw_data_total = json.load(file)
        get_operation_service_pairs(raw_data_total)
        print("Finish main data load !")

    print('Start !')
    # main loop start
    while True:
        print('--------------------------------')
        print(f'time window: {ms2str(start)} ~ {ms2str(end)}')
        # temp
        # if dataLevel == 'trace':
        #     if str(ms2str(start)) not in ['2022-02-25 01:00:00', '2022-02-25 02:00:00', '2022-02-25 03:00:00', '2022-02-25 05:00:00', '2022-02-25 09:00:00', '2022-02-25 11:00:00', '2022-02-25 13:00:00', '2022-02-25 14:00:00', '2022-02-25 17:00:00', '2022-02-25 18:00:00', '2022-02-25 19:00:00']:
        #         start = end
        #         end = start + window_duration
        #         continue
        if ms2str(start) == '2022-05-01 02:12:00' or ms2str(start) == '2022-05-01 00:48:00':    
            # order service 的 trace 断开的原因   2022-05-01 00:48:00 ~ 2022-05-01 00:54:00    2022-05-01 02:06:00 ~ 2022-05-01 02:12:00  2022-05-01 02:12:00 这一段时间的召回率很低（0.30），f1 score 0.45
            # 2022-05-01 01:06:00 ~ 2022-05-01 01:12:00    2022-05-01 01:30:00 ~ 2022-05-01 01:36:00  这一段时间的异常检测完全正确，但是根因定位不准，异常trace的结构完全一样
            print("find it !")
        if intersect_or_not(start1=start, end1=end, start2=request_period_log[2][1], end2=request_period_log[2][2]):
            print("find it !")
            
        timeWindow_count += 1
        abnormal_count = 0
        abnormal_map = {}
        abnormal_seq_map = {}
        normal_seq_map = {}
        STV_map_window = {}
        tid_list = []
        if dataLevel == 'span':
            dataset, raw_data_dict = load_dataset(start, end, dataLevel, 'main')
        elif dataLevel == 'trace':
            dataset, raw_data_dict = load_dataset(start, end, dataLevel, 'main', raw_data_total)
            
        if len(dataset) == 0:
            if start < timestamp(start_str) + (8 * 60 * 60 * 1000):
                start = end
                end = start + window_duration
                continue
            else: 
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
                if use_manual == True:
                    denstream.update_cluster_and_trace_tables()
                sample_label, label_status = denstream.Cluster_AnomalyDetector(sample=np.array(STVector), sample_info=data, data_status='main', manual_labels_list=manual_labels_list)
                # if label_status == 'manual':
                label_map_reCluster[data['trace_id']] = sample_label
            elif AD_method in ['CEDAS_withoutscore', 'CEDAS_withscore']:
                if first_tag:
                    # 1. Initialization
                    sample_label, label_status = cedas.initialization(sample=np.array(STVector), sample_info=data, data_status='main', manual_labels_list=manual_labels_list)
                    # if label_status == 'manual':
                    label_map_reCluster[data['trace_id']] = sample_label
                    first_tag = False
                else:
                    cedas.changed_cluster = None
                    # 2. Update Micro-Clusters
                    sample_label, label_status = cedas.Cluster_AnomalyDetector(sample=np.array(STVector), sample_info=data, data_status='main', manual_labels_list=manual_labels_list)
                    # if label_status == 'manual':
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
                # if data['trace_bool'] == 1:
                    a_pred.append(1)
                    abnormal_map[tid] = True
                    abnormal_count += 1
                    abnormal_seq_map[tid] = data['service_seq']
                else:
                    abnormal_map[tid] = False
                    normal_seq_map[tid] = data['service_seq']
                    a_pred.append(0)

            # label_status
            if label_status == 'manual':
                manual_count += 1

        if AD_method == 'DenStream_withscore':
            # manual_labels_list = denstream.update_cluster_and_trace_tables(manual_labels_list)
            # update cluster table when time window finish
            if use_manual == True:
                denstream.update_cluster_and_trace_tables()
            labels, confidenceScores, sampleRates = denstream.get_labels_confidenceScores_sampleRates(STV_map=STV_map_window, cluster_type=Sample_method)
            # sample_label
            for tid, sample_label in labels.items():
                label_map_reCluster[tid] = sample_label
                if sample_label == 'abnormal':
                    a_pred.append(1)
                    abnormal_map[tid] = True
                    abnormal_count += 1
                    if raw_data_total[tid]['abnormal'] == 0:
                        traceID_list.append(tid)
                else:
                    abnormal_map[tid] = False
                    a_pred.append(0)
            AD_pattern = [micro_cluster for micro_cluster in denstream.p_micro_clusters+denstream.o_micro_clusters if micro_cluster.AD_selected==True]
        elif AD_method == 'CEDAS_withscore':
            manual_labels_list = cedas.update_cluster_and_trace_tables(manual_labels_list)
            labels, confidenceScores, sampleRates = cedas.get_labels_confidenceScores_sampleRates(STV_map=STV_map_window, cluster_type=Sample_method)
            # sample_label
            for tid, sample_label in labels.items():
                label_map_reCluster[tid] = sample_label
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

        if AD_method in ['DenStream_withscore', 'CEDAS_withscore']:
            pattern_IoU = len(set(AD_pattern)&set(rc_pattern)) / len(set(AD_pattern)|set(rc_pattern)) if len(set(AD_pattern)|set(rc_pattern)) != 0 else -1
        else:
            pattern_IoU = -1

        if abnormal_count > 8 and pattern_IoU < 0.5:
            trigger_count += 1

            if AD_method in ['DenStream_withscore', 'CEDAS_withscore']:
                rc_pattern = AD_pattern

            print('********* RCA start *********')

            if AD_method in ['DenStream_withscore', 'CEDAS_withscore']:
                # 在这里对 trace 进行尾采样，若一个微簇/宏观簇的样本数越多，则采样概率低，否则采样概率高
                # 在这里以固定的采样概率对各个簇的 trace 进行采样，不同结构的 trace 应保持相同的采样比例
                # 仅保留采样到的 trace id 即可
                sampled_tid_list = []
                for tid in tid_list:
                    if sampleRates[tid] >= np.random.uniform(0, 1):
                        sampled_tid_list.append(tid)
                if len(sampled_tid_list) != 0:
                    top_list, score_list = rca(RCA_level=RCA_level, start=start, end=end, tid_list=sampled_tid_list, trace_labels=abnormal_map, traces_dict=raw_data_dict, confidenceScores=confidenceScores, dataLevel=dataLevel)
                else:
                    top_list = []
            else:
                top_list, score_list = rca(RCA_level=RCA_level, start=start, end=end, tid_list=tid_list, trace_labels=abnormal_map, traces_dict=raw_data_dict, dataLevel=dataLevel)
            
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
                for idx, root_cause_item in enumerate(request_period_log):
                    # A: start, end    B: root_cause_item[1], root_cause_item[2]
                    # if ((root_cause_item[1]>end and root_cause_item[1]<root_cause_item[2]) or (start<end and root_cause_item[1]>root_cause_item[2]) or (start>end and start<root_cause_item[2])):
                    # if start>=root_cause_item[1] and start<=root_cause_item[2]:
                    if intersect_or_not(start1=start, end1=end, start2=root_cause_item[1], end2=root_cause_item[2]):
                        # if in_rate[str(idx)]==1:
                        #     print("check it !")
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
                if RCA_level == 'operation':
                    for topS in topK_0:
                        candidate_list_0.append(operation_service_dict[topS])
                    for topS in topK_1:
                        candidate_list_1.append(operation_service_dict[topS])
                    for topS in topK_2:
                        candidate_list_2.append(operation_service_dict[topS])
                    for topS in topK_3:
                        candidate_list_3.append(operation_service_dict[topS])
                    for topS in topK_4:
                        candidate_list_4.append(operation_service_dict[topS])
                elif RCA_level == 'service':
                    candidate_list_0 = topK_0
                    candidate_list_1 = topK_1
                    candidate_list_2 = topK_2
                    candidate_list_3 = topK_3
                    candidate_list_4 = topK_4
                if isinstance(chaos_service_list[0], list):    # 一次注入两个故障
                    for service_pair in chaos_service_list:
                        if (service_pair[0] in candidate_list_0) and (service_pair[1] in candidate_list_0):
                            in_topK_0 = True
                        if (service_pair[0] in candidate_list_1) and (service_pair[1] in candidate_list_1):
                            in_topK_1 = True
                        if (service_pair[0] in candidate_list_2) and (service_pair[1] in candidate_list_2):
                            in_topK_2 = True
                        if (service_pair[0] in candidate_list_3) and (service_pair[1] in candidate_list_3):
                            in_topK_3 = True
                        if (service_pair[0] in candidate_list_4) and (service_pair[1] in candidate_list_4):
                            in_topK_4 = True
                else:
                    for service in chaos_service_list:
                        if service in candidate_list_0:
                            in_topK_0 = True
                        if service in candidate_list_1:
                            in_topK_1 = True
                        if service in candidate_list_2:
                            in_topK_2 = True
                        if service in candidate_list_3:
                            in_topK_3 = True
                        if service in candidate_list_4:
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

            # # top_list is not empty
            # if len(top_list) != 0:   
            #     topK = top_list[:K if len(top_list) > K else len(top_list)]
            #     print(f'top-{K} root cause is', topK)
            #     start_hour = time.localtime(start//1000).tm_hour
            #     if dataLevel == 'span':
            #         chaos_service = span_chaos_dict.get(start_hour)
            #     # elif dataLevel == 'trace':
            #     #     chaos_service = trace_chaos_dict.get(start_hour)
            #     elif dataLevel == 'trace':
            #         chaos_service = ''
            #         for root_cause_item in request_period_log:
            #             if start>=root_cause_item[1] and start<=root_cause_item[2]:
            #                 chaos_service = root_cause_item[0][0]
            #                 break

            #     print(f'ground truth root cause is', chaos_service)

            #     # zhoutong add
            #     in_topK = True
            #     candidate_list = []
            #     for topS in topK:
            #         candidate_list += topS.split('/')
            #     if isinstance(chaos_service, list):
            #         for service in chaos_service:
            #             gt_service = service.replace('-', '')[2:]
            #             if gt_service not in candidate_list:
            #                 in_topK = False
            #                 break
            #     else:
            #         gt_service = chaos_service.replace('-', '')[2:]
            #         if gt_service not in candidate_list:
            #             in_topK = False
                        
            # # top_list is empty
            # elif len(top_list) == 0:
            #     in_topK = False
            # r_pred.append(in_topK)
        else:
            rc_pattern = []
            TN += 1
            for root_cause_item in request_period_log:
                if intersect_or_not(start1=start, end1=end, start2=root_cause_item[1], end2=root_cause_item[2]):
                    TN -= 1
                    FN += 1
                    break

        start = end
        end = start + window_duration

        # reCluster ...
        delete_index_candidate = [status[1] for status in all_path.values() if status[0]<path_thres]
        if len(delete_index_candidate) / len(all_path) >= reCluster_thres:
            if AD_method in ['DenStream_withoutscore', 'DenStream_withscore']:
                # if data_status is 'main', all new cluster labels are 'abnormal' and label_status is 'auto'
                # if data_status is 'init', all new cluster labels are 'normal' and label_status is 'auto'
                do_reCluster(cluster_obj=denstream, data_status='main', current_time=start, label_map_reCluster=label_map_reCluster)
                # update cluster table when recluster finish
                if use_manual == True:
                    denstream.update_cluster_and_trace_tables()
            else:
                do_reCluster(cluster_obj=cedas, data_status='main', current_time=start, label_map_reCluster=label_map_reCluster)
        
        # visualization ...
        # if AD_method in ['DenStream_withoutscore', 'DenStream_withscore']:
        #     if len((denstream.p_micro_clusters+denstream.o_micro_clusters)) > 1:
        #         denstream.visualization_tool()
        # else:
        #     if len(cedas.micro_clusters) > 1:
        #         cedas.visualization_tool()
        # main loop end
    
    traceID_fw = open('./manual_traceID.txt', 'w')
    traceID_fw.write(str(traceID_list).replace('[', '').replace(']', '').replace('\'', ''))
    traceID_fw.close()
    print('main loop end')
    

    print('--------------------------------')
    print("Evaluation for TraceStream")
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
        hit_rate1 = r_pred_count_0 / r_true_count
        hit_rate2 = r_pred_count_0 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate 1 is %.5f' % hit_rate1)
        print('RCA hit rate 2 is %.5f' % hit_rate2)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('* * * * * * * *')
    print("Top@{}:".format(K[1]))
    TP = r_pred_count_1
    if TP != 0:
        hit_rate1 = r_pred_count_1 / r_true_count
        hit_rate2 = r_pred_count_1 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate 1 is %.5f' % hit_rate1)
        print('RCA hit rate 2 is %.5f' % hit_rate2)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('* * * * * * * *')
    print("Top@{}:".format(K[2]))
    TP = r_pred_count_2
    if TP != 0:
        hit_rate1 = r_pred_count_2 / r_true_count
        hit_rate2 = r_pred_count_2 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate 1 is %.5f' % hit_rate1)
        print('RCA hit rate 2 is %.5f' % hit_rate2)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('* * * * * * * *')
    print("Top@{}:".format(K[3]))
    TP = r_pred_count_3
    if TP != 0:
        hit_rate1 = r_pred_count_3 / r_true_count
        hit_rate2 = r_pred_count_3 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate 1 is %.5f' % hit_rate1)
        print('RCA hit rate 2 is %.5f' % hit_rate2)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('* * * * * * * *')
    print("Top@{}:".format(K[4]))
    TP = r_pred_count_4
    if TP != 0:
        hit_rate1 = r_pred_count_4 / r_true_count
        hit_rate2 = r_pred_count_4 / trigger_count
        r_acc = (TP + TN)/(TP + FP + TN + FN)
        r_recall = TP/(TP + FN)
        r_prec = TP/(TP + FP)
        print('RCA hit rate 1 is %.5f' % hit_rate1)
        print('RCA hit rate 2 is %.5f' % hit_rate2)
        print('RCA accuracy score is %.5f' % r_acc)
        print('RCA recall score is %.5f' % r_recall)
        print('RCA precision score is %.5f' % r_prec)
    else:
        print('RCA hit rate is 0')
    print('--------------------------------')
    print(r_pred_count_0, r_pred_count_1, r_pred_count_2)
    print("Done !")

if __name__ == '__main__':
    main()