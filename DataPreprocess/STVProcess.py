import os
import json
import numpy as np
import logging
import time
import random
import csv
import click
from enum import Enum
from tqdm import tqdm
from tkinter import _flatten


class NodeType(Enum):
    Req_S = 0
    Req_C = 1
    Res_S = 2
    Res_C = 3
    Log = 4
    DB = 5
    Producer = 6
    Consumer = 7

def deepCopy(source):
    res = []
    for item in source:
        res.append(item)
    return res


def read_process_file(root):
    precesses = []
    f_list = os.listdir(root)
    for i in f_list:
        if os.path.splitext(i)[1] == '.jsons':
            process_path = os.path.join(root, i)
            with open(process_path, 'r') as fin:
                lines = fin.read().strip().split('\n')
                precesses.extend(lines)
    return precesses


def read_csv(file_name):
    # 读取csv至字典
    csvFile = open(file_name, "r")
    reader = csv.reader(csvFile)

    # 建立空字典
    result = []
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        result.append(item[1])
    return result


def replace_service(node):
    global service_list

    node[1] = service_list[node[1]]
    node[2] = service_list[node[2]] if node[2] != -1 else "start"
    return node

# def load_dataset(root):
#     global service_list
#     # global id_url_list

#     service_list = read_csv(os.path.join(root, "id_service.csv"))
#     # id_url_list = read_csv(os.path.join(root, "id_url+temp.csv"))

#     precesses = read_process_file(root)
#     trace_list = list()
#     print("Loading dataset ...")
#     for line in tqdm(precesses):
#         if line.strip() == "":
#             continue
#         trace_list.append(json.loads(line))
#     return trace_list

# def process_one_trace(trace, all_path, all_trace, normal_trace, abnormal_trace):
#     # global id_url_list

#     # 获取一条trace中，对应所有的node节点
#     trace_bool = trace['trace_bool']
#     trace_id = trace['trace_id']
#     node_info = trace['node_info']
#     error_trace_type = trace['error_trace_type']

#     error_type_tid = f"{error_trace_type}_{trace_id}"

#     # 只选取普通节点的req server，以及异步的consumer
#     filter_node = sorted([row for row in node_info if row[4] in [NodeType.Req_S.value, NodeType.Consumer.value]],
#                          key=lambda x: (x[3]))
#     # 排序，首先按照start_time-rt，其次按照str保证call 前后顺序，即就是一条trace中，保证其path中的元素都是一样的
#     filter_node = sorted(list(map(replace_service, [node for node in filter_node])), key=lambda x: (x[3], x[1]))
#     single_trace_results = []
#     # root_url = ""

#     # "url" 0, "service" 1, "parent_service" 2, "rt" 3, "type" 4, "span_id" 5, "u_id" 6
#     for node in filter_node:
#         url = int(node[0])
#         service = node[1]
#         parent_service = node[2]

#         single_call_path = {"service": service, "url": url, "rt": 0, "paths": []}

#         if parent_service == "start":
#             single_call_path["rt"] = node[3]

#             start_time = 0
#             call_path = "start_" + service
#             time_path_rt = (start_time, call_path)
#             single_call_path["paths"].append(time_path_rt)
#             # root_url = call_path + "_" + id_url_list[url]
#         else:
#             # 计算 rt ，在 server 端，res_s - req_s。span_id相同，type 是 res_s
#             res_s = [row for row in node_info if row[5] == node[5] and row[4] == NodeType.Res_S.value]
#             single_call_path["rt"] = res_s[0][3] - node[3]

#             call_path = parent_service + "_" + service

#             # 构造新加入的 path 元组。需要利用 start_time -base_time 来进行排序，根据相对顺序就可以进行排序了
#             start_time = node[3]
#             time_path = (start_time, call_path)

#             # 将当前调用元组 () 与之前的进行拼接，同时path内部元素按照str排序，只要保证path元素和前面的service一样，那么必是同一个调用
#             if len(single_trace_results) == 0:
#                 single_call_path["paths"].append(time_path)
#                 continue
#             previous = single_trace_results[-1]
#             previous_paths = previous["paths"]
#             paths = deepCopy(previous_paths)

#             # paths:[(time,path),(time,path),(time,path)]
#             paths.append(time_path)
#             path_sort = sorted(paths, key=lambda x: (x[1].lower()))
#             single_call_path["paths"].extend(path_sort)

#         # 将排好序的path放入unique_path中
#         # ('ts-order-service', ('start_ts-cancel-service', 'ts-cancel-service_ts-order-service'))
#         _, path_list = zip(*single_call_path["paths"])
#         if (single_call_path["service"], path_list) not in all_path:
#             all_path.append((single_call_path["service"], path_list))
#         single_trace_results.append(single_call_path)

#         '''
#         if root_url in url_trace.keys():
#             url_path = url_trace[root_url]
#             url_path.append((single_call_path["service"], path_list, len(filter_node)))
#             url_trace[root_url] = list(set(url_path))
#         else:
#             url_trace[root_url] = [(single_call_path["service"], path_list, len(filter_node))]
#         '''

#     if trace_bool:
#         normal_trace[error_type_tid] = single_trace_results
#     else:
#         abnormal_trace[error_type_tid] = single_trace_results
#     all_trace[error_type_tid] = single_trace_results
#     return all_path, all_trace, normal_trace, abnormal_trace
    

def load_data(root, logger):
    start = time.time()
    global service_list

    service_list = read_csv(os.path.join(root, "id_service.csv"))
    id_url_list = read_csv(os.path.join(root, "id_url+temp.csv"))

    precesses = read_process_file(root)
    trace_list = list()
    for line in precesses:
        if line.strip() == "":
            continue
        trace_list.append(json.loads(line))
    logger.info(
        "Finish load trace file from %s, total trace count [ %s ], use time [ %.2f ] " % (root, (len(precesses)),
                                                                                          (time.time() - start)))
    all_path = []
    normal_trace = {}
    abnormal_trace = {}
    url_trace = {}

    for trace in tqdm(trace_list, desc="trace : "):
        # 获取一条trace中，对应所有的node节点
        trace_bool = trace['trace_bool']
        trace_id = trace['trace_id']
        node_info = trace['node_info']
        error_trace_type = trace['error_trace_type']

        error_type_tid = f"{error_trace_type}_{trace_id}"

        # 只选取普通节点的req server，以及异步的consumer
        filter_node = sorted([row for row in node_info if row[4] in [NodeType.Req_S.value, NodeType.Consumer.value]],
                             key=lambda x: (x[3]))
        # 排序，首先按照start_time-rt，其次按照str保证call 前后顺序，即就是一条trace中，保证其path中的元素都是一样的
        filter_node = sorted(list(map(replace_service, [node for node in filter_node])), key=lambda x: (x[3], x[1]))
        single_trace_results = []
        root_url = ""

        # "url" 0, "service" 1, "parent_service" 2, "rt" 3, "type" 4, "span_id" 5, "u_id" 6
        for node in filter_node:
            url = int(node[0])
            service = node[1]
            parent_service = node[2]

            single_call_path = {"service": service, "url": url, "rt": 0, "paths": []}

            if parent_service == "start":
                single_call_path["rt"] = node[3]

                start_time = 0
                call_path = "start_" + service
                time_path_rt = (start_time, call_path)
                single_call_path["paths"].append(time_path_rt)
                root_url = call_path + "_" + id_url_list[url]
            else:
                # 计算 rt ，在 server 端，res_s - req_s。span_id相同，type 是 res_s
                res_s = [row for row in node_info if row[5] == node[5] and row[4] == NodeType.Res_S.value]
                single_call_path["rt"] = res_s[0][3] - node[3]

                call_path = parent_service + "_" + service

                # 构造新加入的 path 元组。需要利用 start_time -base_time 来进行排序，根据相对顺序就可以进行排序了
                start_time = node[3]
                time_path = (start_time, call_path)

                # 将当前调用元组 () 与之前的进行拼接，同时path内部元素按照str排序，只要保证path元素和前面的service一样，那么必是同一个调用
                if len(single_trace_results) == 0:
                    single_call_path["paths"].append(time_path)
                    continue
                previous = single_trace_results[-1]
                previous_paths = previous["paths"]
                paths = deepCopy(previous_paths)

                # paths:[(time,path),(time,path),(time,path)]
                paths.append(time_path)
                path_sort = sorted(paths, key=lambda x: (x[1].lower()))
                single_call_path["paths"].extend(path_sort)

            # 将排好序的path放入unique_path中
            # ('ts-order-service', ('start_ts-cancel-service', 'ts-cancel-service_ts-order-service'))
            _, path_list = zip(*single_call_path["paths"])
            all_path.append((single_call_path["service"], path_list))
            single_trace_results.append(single_call_path)

            if root_url in url_trace.keys():
                url_path = url_trace[root_url]
                url_path.append((single_call_path["service"], path_list, len(filter_node)))
                url_trace[root_url] = list(set(url_path))
            else:
                url_trace[root_url] = [(single_call_path["service"], path_list, len(filter_node))]

        if trace_bool:
            normal_trace[error_type_tid] = single_trace_results
        else:
            abnormal_trace[error_type_tid] = single_trace_results
        # logger.info("Trace %s has node %s, path count %s" % (trace_id, len(filter_node), len(single_trace_results)))
        # logger.info("Total path count %s, total unique path count %s" % (len(all_path), len(set(all_path))))

        # if (len(single_trace_results)) > 30:
        #     logger.info("Root call path %s" % (root_url))
        # if root_call_path == "start_ts-travel-service":
        #     origin = len(set(travel_trace))
        #     travel_trace.extend(set(trace_path))
        #     now = len(set(travel_trace))
        #     logger.info("unique total %s, simple %s, new %s" % (
        #         (now), len(trace_path), (now - origin)))

    unique = 0
    for key, value in url_trace.items():
        length = len(value)
        unique += length
        if length > 50:
            logger.info("length > 50 url %s, length %s" % (key, len(value)))
    logger.info("url total %s" % unique)
    unique_path = list(set(all_path))    # !!!!!!!!! 检查一下 set list 变化会不会影响元素顺序，set 操作是不是仅保留第一次出现的元素
    logger.info(
        "Finish prepare trace data,normal_trace [ %s ],abnormal_trace [ %s ],unique_path [ %s ] use time [ %s ]" % (
            len(normal_trace), len(abnormal_trace), len(unique_path), time.time() - start))
    return normal_trace, abnormal_trace, unique_path

# def embedding_to_vector(trace_results, unique_path):
#     dim = len(unique_path)
#     trace_vector = []
#     for trace_id, results in trace_results.items():
#         myMat = -1*np.ones(dim)    # 需要换成 -1
#         for result in results:
#             rt = result["rt"]
#             service = result["service"]
#             _, path_list = zip(*result["paths"])
#             index = unique_path.index((service, path_list))
#             myMat[index] = rt
#         vector_list = myMat.tolist()
#         trace_vector.append(vector_list)
#     return trace_vector

def embedding(trace_results, unique_path, logger):
    start = time.time()

    dim = len(unique_path)
    trace_vector = []
    for trace_id, results in trace_results.items():
        index_rt = []
        myMat = np.zeros(dim)    # 需要换成 -1
        for result in results:
            rt = result["rt"]
            service = result["service"]
            _, path_list = zip(*result["paths"])
            index = unique_path.index((service, path_list))
            index_rt.append((index, rt))
            myMat[index] = rt
        vector_list = myMat.tolist()
        vector_str = ",".join(str(i) for i in vector_list)
        data = trace_id + ":" + vector_str
        trace_vector.append(data)
    logger.info("Finish embedding trace data, the dim is [ %s ],Total trace count [ %s ] use time [ %s ]," % (
        dim, len(trace_results), (time.time() - start)))
    return trace_vector


def prepare_data_to_file(total_normal, total_abnormal, output, ratio, need_abnormal=False, logger=None):
    random.shuffle(total_normal)
    random.shuffle(total_abnormal)

    total_normal_num = len(total_normal)
    total_abnormal_num = len(total_abnormal)
    train_normal_num = int(total_normal_num * ratio[0] / sum(ratio))
    if need_abnormal:
        train_abnormal_num = int(total_abnormal_num * ratio[0] / sum(ratio))
    else:
        train_abnormal_num = 0

    pre_train = deepCopy(total_normal[:train_normal_num])
    pre_train.extend(deepCopy(total_abnormal[:train_abnormal_num]))

    pre_test_normal = deepCopy(total_normal[train_normal_num:])
    # test_normal_num = len(pre_test_normal)
    #
    # rest = total_abnormal_num - train_abnormal_num
    # test_abnormal_num = test_normal_num if test_normal_num < rest else rest

    pre_test_abnormal = deepCopy(total_abnormal[train_abnormal_num:])
    pre_test_count = len(pre_test_normal) + len(pre_test_abnormal)

    logger.info(
        'Train: %d, Test: %d' % (len(pre_train), pre_test_count))
    logger.info('Train: %d, normal: %d, abnormal: %d' % (len(pre_train), train_normal_num, train_abnormal_num))
    logger.info('Test : %d, normal: %d, abnormal: %d' % (
        len(pre_test_normal) + len(pre_test_abnormal), len(pre_test_normal), len(pre_test_abnormal)))

    random.shuffle(pre_train)

    save_train_path = os.path.join(output, "train")
    save_test_normal_path = os.path.join(output, "test_normal")
    save_test_abnormal_path = os.path.join(output, "test_abnormal")
    save_vector_to_file(pre_train, save_train_path)
    save_vector_to_file(pre_test_normal, save_test_normal_path)
    save_vector_to_file(pre_test_abnormal, save_test_abnormal_path)


def save_vector_to_file(trace_vector, save_path):
    with open(save_path, 'w') as f:
        for line in trace_vector:
            f.write(line + '\n')
        f.close()






   




import json
import sys

import numpy as np
from tqdm import tqdm
from .SpanProcess import preprocess_span

same_seq_test_list = list()
service_rt_dict = dict()    # {service1: [rt1, rt2, ...], service2: [rt1, rt2, ...]}
service_rt_normalize_dict = dict()    # {service1: [mean, std, max, min], service2: [mean, std, max, min]}
operation_rt_dict = dict()    # {operation1: [rt1, rt2, ...], operation2: [rt1, rt2, ...]}
operation_rt_normalize_dict = dict()    # {operation1: [mean, std], operation2: [mean, std], ...}

def z_score(x: float, mean: float, std: float) -> float:
    """
    z-score normalize funciton
    """
    return (float(x) - float(mean)) / float(std)


def min_max(x: float, min: float, max: float) -> float:
    """
    min-max normalize funciton
    """
    return (float(x) - float(min)) / (float(max) - float(min))

def load_dataset(start, end, dataLevel, stage, raw_data_total=None):    # stage: 'main', 'init' 对'init'计算所有operation的均值方差再求归一化后结果 对'main'的每个operation求归一化后结果
    trace_list = list()
    raw_data = dict()

    if dataLevel == 'span':
        raw_data = preprocess_span(start, end, stage)
    elif dataLevel == 'trace':           
        for trace_id, trace in sorted(raw_data_total.items(), key = lambda item: item[1]['edges']['0'][0]['startTime']):
            if trace['edges']['0'][0]['startTime']>=end:
                break
            if trace['edges']['0'][0]['startTime']>=start and trace['edges']['0'][0]['startTime']<end:
                raw_data[trace_id] = trace

    # print('getting trace data (api and time seq) ... 1')
    for trace_id, trace in sorted(raw_data.items(), key = lambda i: i[1]['edges']['0'][0]['startTime']):
        service_seq = ['start']
        spans = []
        for span in trace['edges'].values():
            spans.extend(span)
        spans = sorted(spans, key=lambda span: span['startTime'])
        service_seq.extend([span['service'] for span in spans])
        
        
        # time_seq = []
        # break_label = False
        # for span in spans:
        #     if span['service'] in service_rt_normalize_dict:
        #         time_seq.append(z_score(x=float(span['rawDuration']), mean=service_rt_normalize_dict[span['service']][0], std=service_rt_normalize_dict[span['service']][1]))
        #     else:
        #         break_label = True
        #         raw_data.pop(trace_id)
        #         break
        # if break_label == True:
        #     continue

        
        # normalize1 (z-score)
        time_seq = [z_score(x=float(span['rawDuration']), mean=service_rt_normalize_dict[span['service']][0], std=service_rt_normalize_dict[span['service']][1]) for span in spans]    
        # normalize2 (min-max)
        # time_seq = [min_max(x=float(span['rawDuration']), min=service_rt_normalize_dict[span['service']][3], max=service_rt_normalize_dict[span['service']][2]) for span in spans]    
        # original
        # time_seq = [span['rawDuration'] for span in spans]
        isError_seq = [span['isError'] for span in spans]
        time_stamp = trace['edges']['0'][0]['startTime']
        trace_list.append({'trace_id': trace_id, 'service_seq': service_seq, 'time_seq': time_seq, 'isError_seq': isError_seq, 'time_stamp': time_stamp, 'trace_bool': trace['abnormal']})
        
        # test
        # if service_seq==['start', 'ts-order-service']: 
        #    same_seq_test_list.append({'service_seq': service_seq, 'time_seq': time_seq, 'time_stamp': time_stamp, 'trace_bool': trace['abnormal']})
    # with open(r'G:/workspace/TraceCluster/newData/preprocessed_old/abnormal.json', 'r') as file_2:
    #     raw_data = json.load(file_2)
    #     print('getting trace data (api and time seq) ... 2')
    #     for trace_id, trace in tqdm(raw_data.items()):
    #         service_seq = ['start']
    #         spans = []
    #         for span in trace['edges'].values():
    #             spans.extend(span)
    #         spans = sorted(spans, key=lambda span: span['startTime'])
    #         service_seq.extend([span['service'] for span in spans])
    #         time_seq = [span['rawDuration'] for span in spans]
    #         trace_list.append({'trace_id': trace_id, 'service_seq': service_seq, 'time_seq': time_seq, 'trace_bool': trace['abnormal']})
    return trace_list, raw_data


def load_raw_dataset(start, end, raw_data_total):
    raw_data = dict()
    for trace_id, trace in sorted(raw_data_total.items(), key = lambda item: item[1]['edges']['0'][0]['startTime']):
        if trace['edges']['0'][0]['startTime']>=end:
            break
        if trace['edges']['0'][0]['startTime']>=start and trace['edges']['0'][0]['startTime']<end:
            raw_data[trace_id] = trace
    return raw_data

def get_service_slo(raw_data):
    for trace_id, trace in raw_data.items():
        for spans in trace['edges'].values():
            for span in spans:
                if span['service'] not in service_rt_dict.keys():
                    service_rt_dict[span['service']] = [span['rawDuration']]
                else:
                    service_rt_dict[span['service']].append(span['rawDuration'])
    for service, rt_list in service_rt_dict.items():
        service_rt_normalize_dict[service] = [np.mean(rt_list), np.std(rt_list), np.max(rt_list), np.min(rt_list)]

def get_operation_slo(raw_data):
    for trace_id, trace in raw_data.items():
        for spans in trace['edges'].values():
            for span in spans:
                if span['operation'] not in operation_rt_dict.keys():
                    operation_rt_dict[span['operation']] = [span['rawDuration']]
                else:
                    operation_rt_dict[span['operation']].append(span['rawDuration'])
    for operation, rt_list in operation_rt_dict.items():
        operation_rt_normalize_dict[operation] = [round(np.mean(rt_list) / 1000.0, 4), round(np.std(rt_list) / 1000.0, 4)]            
    return operation_rt_normalize_dict


def get_operation_duration_data(raw_data):
    operation_dict = dict()
    for trace_id, trace in raw_data.items():
        operation_dict[trace_id] = dict()
        operation_dict[trace_id]['duration'] = 0
        for spans in trace['edges'].values():
            for span in spans:
                if span['operation'] not in operation_dict[trace_id].keys():
                    operation_dict[trace_id][span['operation']] = 1
                else:
                    operation_dict[trace_id][span['operation']] += 1
                operation_dict[trace_id]['duration'] += span['rawDuration']
    return operation_dict


def process_one_trace(trace, unique_path):
    for i in range(1, len(trace['service_seq'])):
        if '->'.join(trace['service_seq'][:i + 1]) not in unique_path.keys():
            # new path
            unique_path['->'.join(trace['service_seq'][:i + 1])] = [1, len(unique_path)]    # [energy, index]
            # unique_path.append('->'.join(trace['service_seq'][:i + 1]))
        else:
            # old path
            unique_path['->'.join(trace['service_seq'][:i + 1])][0] = 1    # energy 重新置一
    return unique_path


def embedding_to_vector(trace, unique_path):
    length = len(unique_path)
    # vector_list = [0 for i in range(length)]
    vector_list = [[0, 0] for i in range(length)]    # vector_list: [[rt1, isError1], [rt2, isError2]]
    for i in range(1, len(trace['service_seq'])):
        # vector_list[unique_path['->'.join(trace['service_seq'][:i + 1])][1]] = trace['time_seq'][i-1] 

        vector_list[unique_path['->'.join(trace['service_seq'][:i + 1])][1]] = [trace['time_seq'][i-1], 0 if trace['isError_seq'][i-1]==False else 1] 
        # vector_list[unique_path.index('->'.join(trace['service_seq'][:i + 1]))] = trace['time_seq'][i-1]
        # for isError_item in trace['isError_seq'][:i]:
        #     if isError_item == True:
        #         vector_list[unique_path['->'.join(trace['service_seq'][:i + 1])][1]][1] = 1
        #         break
    return list(_flatten(vector_list))


def check_match():
    # request_period_log = [(['ts-ticketinfo-service'], 1650641167424, 1650641474258), (['ts-order-service'], 1650641574378, 1650641888464), (['ts-route-service'], 1650641988585, 1650642301332), (['ts-auth-service'], 1650642401453, 1650642734489), (['ts-auth-service'], 1650643240663, 1650643564678), (['ts-ticketinfo-service'], 1650644475959, 1650644780866), (['ts-order-service'], 1650644880993, 1650645186599), (['ts-route-service'], 1650645688256, 1650646003680), (['ts-user-service'], 1650646103797, 1650646403929), (['ts-order-service'], 1650646504048, 1650646810985), (['ts-route-service'], 1650646911106, 1650647229957), (['ts-order-service'], 1650647732560, 1650648038136), (['ts-route-service'], 1650648138255, 1650648476697), (['ts-order-service'], 1650649385278, 1650649690202), (['ts-route-service'], 1650649790322, 1650650108198), (['ts-user-service'], 1650650208317, 1650650510867), (['ts-order-service'], 1650650610988, 1650650915668), (['ts-route-service'], 1650651886870, 1650652192271), (['ts-ticketinfo-service'], 1650653494728, 1650653797611), (['ts-order-service'], 1650654300889, 1650654605778), (['ts-route-service'], 1650654705891, 1650655022755), (['ts-user-service'], 1650655122865, 1650655424565), (['ts-travel-service'], 1650655524681, 1650655828606), (['ts-ticketinfo-service'], 1650655928755, 1650656236254), (['ts-order-service'], 1650656336374, 1650656641766), (['ts-ticketinfo-service'], 1650656741888, 1650657044303), (['ts-order-service'], 1650657144418, 1650657448876), (['ts-order-service'], 1650657548999, 1650657854457), (['ts-order-service'], 1650658788037, 1650659091412), (['ts-route-service'], 1650659997502, 1650660373783), (['ts-order-service'], 1650660473901, 1650660775220)]
    # request_period_log = [(['ts-station-service', 'ts-travel-plan-service'], 1650799190100, 1650799503849), (['ts-travel-plan-service', 'ts-ticketinfo-service'], 1650799694546, 1650800003505), (['ts-travel-plan-service', 'ts-user-service'], 1650800194225, 1650800499167), (['ts-route-service', 'ts-user-service'], 1650800689886, 1650800990422), (['ts-order-service', 'ts-route-service'], 1650801181181, 1650801487521), (['ts-travel-service', 'ts-ticketinfo-service'], 1650801678227, 1650801982366), (['ts-travel-plan-service', 'ts-station-service'], 1650802295697, 1650802603234), (['ts-station-service', 'ts-route-service'], 1650802793966, 1650803098596), (['ts-user-service', 'ts-order-service'], 1650803289302, 1650803594537), (['ts-order-service', 'ts-basic-service'], 1650803785275, 1650804090999), (['ts-ticketinfo-service', 'ts-travel-plan-service'], 1650804281703, 1650804587126), (['ts-ticketinfo-service', 'ts-travel-service'], 1650804777821, 1650805078830), (['ts-travel-service', 'ts-basic-service'], 1650805269546, 1650805573167), (['ts-route-service', 'ts-order-service'], 1650805763890, 1650806067321), (['ts-route-service', 'ts-ticketinfo-service'], 1650806258018, 1650806563350), (['ts-basic-service', 'ts-route-service'], 1650806754077, 1650807058218), (['ts-ticketinfo-service', 'ts-travel-service'], 1650807248906, 1650807554234), (['ts-basic-service', 'ts-basic-service'], 1650807744931, 1650808051765), (['ts-travel-service', 'ts-order-service'], 1650808242448, 1650808600766), (['ts-user-service', 'ts-station-service'], 1650808791455, 1650809097474), (['ts-basic-service', 'ts-travel-service'], 1650809288169, 1650809591884), (['ts-station-service', 'ts-user-service'], 1650809782583, 1650810083310), (['ts-user-service', 'ts-station-service'], 1650810273996, 1650810580125), (['ts-order-service', 'ts-travel-plan-service'], 1650810770817, 1650811073429), (['ts-ticketinfo-service', 'ts-travel-service'], 1650811264128, 1650811569038), (['ts-user-service', 'ts-ticketinfo-service'], 1650811759737, 1650812063955), (['ts-station-service', 'ts-ticketinfo-service'], 1650812254662, 1650812559676), (['ts-ticketinfo-service', 'ts-travel-service'], 1650812750379, 1650813056492), (['ts-route-service', 'ts-order-service'], 1650813247180, 1650813553395), (['ts-route-service', 'ts-user-service'], 1650813744119, 1650814047621), (['ts-order-service', 'ts-station-service'], 1650814238319, 1650814544734), (['ts-ticketinfo-service', 'ts-basic-service'], 1650814735430, 1650815038854), (['ts-order-service', 'ts-route-service'], 1650815229528, 1650815535653), (['ts-basic-service', 'ts-ticketinfo-service'], 1650815726335, 1650816033550), (['ts-travel-service', 'ts-order-service'], 1650816224229, 1650816528146), (['ts-user-service', 'ts-travel-service'], 1650816718855, 1650817024253), (['ts-travel-plan-service', 'ts-travel-plan-service'], 1650817214950, 1650817517647), (['ts-travel-service', 'ts-basic-service'], 1650817708339, 1650818013343), (['ts-user-service', 'ts-user-service'], 1650818204021, 1650818508128), (['ts-travel-plan-service', 'ts-travel-plan-service'], 1650818698827, 1650818999444), (['ts-travel-service', 'ts-user-service'], 1650819190140, 1650819495239), (['ts-station-service', 'ts-station-service'], 1650819685930, 1650819989940), (['ts-basic-service', 'ts-station-service'], 1650820180622, 1650820482333), (['ts-station-service', 'ts-route-service'], 1650820673033, 1650820977730), (['ts-order-service', 'ts-route-service'], 1650821168406, 1650821472008), (['ts-travel-plan-service', 'ts-basic-service'], 1650821662699, 1650821968898), (['ts-basic-service', 'ts-order-service'], 1650822159599, 1650822462807), (['ts-route-service', 'ts-travel-plan-service'], 1650822653528, 1650822953929), (['ts-station-service', 'ts-user-service'], 1650823144636, 1650823444945), (['ts-travel-plan-service', 'ts-travel-plan-service'], 1650823635651, 1650823938547)]

    # request_period_log = [(['ts-route-service'], 1650254059896, 1650254361387), (['ts-travel-service'], 1650254551823, 1650254853456), (['ts-travel-service'], 1650255043883, 1650255344428), (['ts-travel-service'], 1650255534831, 1650255851210), (['ts-station-service'], 1650256151142, 1650256452270), (['ts-basic-service'], 1650256642674, 1650256944234), (['ts-ticketinfo-service'], 1650257134636, 1650257435240), (['ts-user-service'], 1650257625637, 1650257927939), (['ts-order-service'], 1650258118339, 1650258421608), (['ts-order-service'], 1650258611989, 1650258912546), (['ts-order-service'], 1650259102938, 1650259403142), (['ts-basic-service'], 1650259593530, 1650259896864), (['ts-route-service'], 1650260087260, 1650260388746), (['ts-user-service'], 1650260579154, 1650260881122), (['ts-station-service'], 1650261071509, 1650261371783), (['ts-ticketinfo-service'], 1650261562192, 1650261865276), (['ts-travel-plan-service'], 1650262055678, 1650262358656), (['ts-travel-plan-service'], 1650262549050, 1650262852081), (['ts-user-service'], 1650263042497, 1650263343295), (['ts-basic-service'], 1650263533699, 1650263836890), (['ts-ticketinfo-service'], 1650264027284, 1650264328269), (['ts-station-service'], 1650264518660, 1650264821500), (['ts-travel-plan-service'], 1650265011904, 1650265312147), (['ts-route-service'], 1650265502560, 1650265803476), (['ts-travel-service'], 1650265993869, 1650266305823),
    #                       (['ts-route-service'], 1650266496226, 1650266796479), (['ts-basic-service'], 1650266986888, 1650267287653), (['ts-travel-service'], 1650267478060, 1650267779310), (['ts-user-service'], 1650267969704, 1650268271204), (['ts-ticketinfo-service'], 1650268461603, 1650268763257), (['ts-travel-service'], 1650268953663, 1650269257054), (['ts-order-service'], 1650269447456, 1650269749062), (['ts-station-service'], 1650269939458, 1650270241764), (['ts-user-service'], 1650270432172, 1650270732294), (['ts-order-service'], 1650270922698, 1650271225238), (['ts-travel-plan-service'], 1650271415635, 1650271715726), (['ts-order-service'], 1650271906150, 1650272206462), (['ts-basic-service'], 1650272396851, 1650272699948), (['ts-basic-service'], 1650272890362, 1650273192473), (['ts-route-service'], 1650273382897, 1650273684256), (['ts-station-service'], 1650273874658, 1650274177836), (['ts-ticketinfo-service'], 1650274368244, 1650274668476), (['ts-station-service'], 1650274858860, 1650275159546), (['ts-route-service'], 1650275349948, 1650275652839), (['ts-travel-plan-service'], 1650275843231, 1650276144522), (['ts-user-service'], 1650276334921, 1650276636243), (['ts-ticketinfo-service'], 1650276826642, 1650277128245), (['ts-travel-plan-service'], 1650277318638, 1650277619725), (['ts-order-service'], 1650277810111, 1650278110838), (['ts-travel-service'], 1650278301231, 1650278626536)]
    # request_period_log = [(['ts-route-service'], 1650978142584, 1650978445305), (['ts-travel-service'], 1650978635707, 1650978961865), (['ts-travel-service'], 1650979152262, 1650979460973), (['ts-travel-service'], 1650979651364, 1650980004858), (['ts-station-service'], 1650980195250, 1650980499254), (['ts-basic-service'], 1650980689677, 1650980994204), (['ts-ticketinfo-service'], 1650981184602, 1650981493825), (['ts-user-service'], 1650981684256, 1650981992263), (['ts-order-service'], 1650982182661, 1650982483576), (['ts-order-service'], 1650982673982, 1650982979787), (['ts-order-service'], 1650983170180, 1650983480798), (['ts-basic-service'], 1650983671199, 1650983978223), (['ts-route-service'], 1650984168626, 1650984472640), (['ts-user-service'], 1650984663050, 1650984964054), (['ts-station-service'], 1650985154468, 1650985458274), (['ts-ticketinfo-service'], 1650985648682, 1650985956000), (['ts-travel-plan-service'], 1650986146416, 1650986454634), (['ts-travel-plan-service'], 1650986645044, 1650986947542), (['ts-user-service'], 1650987137950, 1650987441362), (['ts-basic-service'], 1650987631768, 1650987937475), (['ts-ticketinfo-service'], 1650988127884, 1650988427980), (['ts-station-service'], 1650988618388, 1650988920685), (['ts-route-service'], 1650989111093, 1650989411993), (['ts-route-service'], 1650989602376, 1650989903480), (['ts-travel-service'], 1650990093885, 1650990399800),
    #                   (['ts-route-service'], 1650990590187, 1650990890385), (['ts-basic-service'], 1650991080781, 1650991382580), (['ts-travel-service'], 1650991572990, 1650991880294), (['ts-user-service'], 1650992070708, 1650992371601), (['ts-ticketinfo-service'], 1650992561997, 1650992863287), (['ts-travel-service'], 1650993053718, 1650993357412), (['ts-order-service'], 1650993547806, 1650993847907), (['ts-station-service'], 1650994038392, 1650994338497), (['ts-user-service'], 1650994528885, 1650994833185), (['ts-order-service'], 1650995023587, 1650995330885), (['ts-travel-plan-service'], 1650995521277, 1650995822762), (['ts-order-service'], 1650996013157, 1650996318547), (['ts-basic-service'], 1650996508925, 1650996812521), (['ts-basic-service'], 1650997002915, 1650997306422), (['ts-route-service'], 1650997496827, 1650997802820), (['ts-station-service'], 1650997993213, 1650998300522), (['ts-ticketinfo-service'], 1650998490929, 1650998799331), (['ts-station-service'], 1650998989720, 1650999297412), (['ts-route-service'], 1650999487855, 1650999788542), (['ts-travel-plan-service'], 1650999978950, 1651000281637), (['ts-user-service'], 1651000472026, 1651000776508), (['ts-ticketinfo-service'], 1651000966967, 1651001273564), (['ts-route-service'], 1651001463964, 1651001772659), (['ts-order-service'], 1651001963058, 1651002263129), (['ts-travel-service'], 1651002453539, 1651002760436)]
    # request_period_log = [(['ts-route-service'], 1650978142584, 1650978445305), (['ts-travel-service'], 1650978635707, 1650978961865), (['ts-travel-service'], 1650979152262, 1650979460973), (['ts-travel-service'], 1650979651364, 1650980004858), (['ts-station-service'], 1650980195250, 1650980499254), (['ts-basic-service'], 1650980689677, 1650980994204), (['ts-ticketinfo-service'], 1650981184602, 1650981493825), (['ts-user-service'], 1650981684256, 1650981992263), (['ts-order-service'], 1650982182661, 1650982483576), (['ts-order-service'], 1650982673982, 1650982979787), (['ts-order-service'], 1650983170180, 1650983480798), (['ts-basic-service'], 1650983671199, 1650983978223), (['ts-route-service'], 1650984168626, 1650984472640), (['ts-user-service'], 1650984663050, 1650984964054), (['ts-station-service'], 1650985154468, 1650985458274), (['ts-ticketinfo-service'], 1650985648682, 1650985956000), (['ts-travel-plan-service'], 1650986146416, 1650986454634), (['ts-travel-plan-service'], 1650986645044, 1650986947542), (['ts-user-service'], 1650987137950, 1650987441362), (['ts-basic-service'], 1650987631768, 1650987937475), (['ts-ticketinfo-service'], 1650988127884, 1650988427980), (['ts-station-service'], 1650988618388, 1650988920685), (['ts-route-service'], 1650989111093, 1650989411993), (['ts-route-service'], 1650989602376, 1650989903480), (['ts-travel-service'], 1650990093885, 1650990399800),
    #                     (['ts-route-service'], 1650990590187, 1650990890385), (['ts-basic-service'], 1650991080781, 1650991382580), (['ts-travel-service'], 1650991572990, 1650991880294), (['ts-user-service'], 1650992070708, 1650992371601), (['ts-ticketinfo-service'], 1650992561997, 1650992863287), (['ts-travel-service'], 1650993053718, 1650993357412), (['ts-order-service'], 1650993547806, 1650993847907), (['ts-station-service'], 1650994038392, 1650994338497), (['ts-user-service'], 1650994528885, 1650994833185), (['ts-order-service'], 1650995023587, 1650995330885), (['ts-travel-plan-service'], 1650995521277, 1650995822762), (['ts-order-service'], 1650996013157, 1650996318547), (['ts-basic-service'], 1650996508925, 1650996812521), (['ts-basic-service'], 1650997002915, 1650997306422), (['ts-route-service'], 1650997496827, 1650997802820), (['ts-station-service'], 1650997993213, 1650998300522), (['ts-ticketinfo-service'], 1650998490929, 1650998799331), (['ts-station-service'], 1650998989720, 1650999297412), (['ts-route-service'], 1650999487855, 1650999788542), (['ts-travel-plan-service'], 1650999978950, 1651000281637), (['ts-user-service'], 1651000472026, 1651000776508), (['ts-ticketinfo-service'], 1651000966967, 1651001273564), (['ts-route-service'], 1651001463964, 1651001772659), (['ts-order-service'], 1651001963058, 1651002263129), (['ts-travel-service'], 1651002453539, 1651002760436)]
    




    # 2 ab
    # request_period_log = [(['ts-station-service', 'ts-route-service'], 1651118425739, 1651118758799), (['ts-travel-plan-service', 'ts-ticketinfo-service'], 1651118949487, 1651119269011), (['ts-travel-plan-service', 'ts-user-service'], 1651119459694, 1651119767415), (['ts-route-service', 'ts-user-service'], 1651119958119, 1651120264839), (['ts-order-service', 'ts-route-service'], 1651120455528, 1651120757346), (['ts-travel-service', 'ts-ticketinfo-service'], 1651120948033, 1651121254747), (['ts-route-service', 'ts-station-service'], 1651121445461, 1651121750077), (['ts-station-service', 'ts-route-service'], 1651121940771, 1651122246990), (['ts-user-service', 'ts-order-service'], 1651122437709, 1651122745710), (['ts-order-service', 'ts-basic-service'], 1651122936404, 1651123240019), (['ts-ticketinfo-service', 'ts-travel-plan-service'], 1651123430717, 1651123739127), (['ts-ticketinfo-service', 'ts-travel-service'], 1651123929854, 1651124235665), (['ts-travel-service', 'ts-basic-service'], 1651124426365, 1651124734072), (['ts-route-service', 'ts-order-service'], 1651124924754, 1651125230573), (['ts-route-service', 'ts-ticketinfo-service'], 1651125421286, 1651125732309), (['ts-basic-service', 'ts-route-service'], 1651125923003, 1651126223696), (['ts-ticketinfo-service', 'ts-travel-service'], 1651126414387, 1651126721790), (['ts-basic-service', 'ts-basic-service'], 1651126912475, 1651127218490), (['ts-travel-service', 'ts-order-service'], 1651127409181, 1651127714488), (['ts-user-service', 'ts-station-service'], 1651127905188, 1651128207091), (['ts-basic-service', 'ts-travel-service'], 1651128397794, 1651128703402), (['ts-station-service', 'ts-user-service'], 1651128894116, 1651129201121), (['ts-user-service', 'ts-station-service'], 1651129391849, 1651129693049), (['ts-order-service', 'ts-travel-plan-service'], 1651129883749, 1651130192165), (['ts-ticketinfo-service', 'ts-travel-service'], 1651130382913, 1651130687516), (['ts-user-service', 'ts-ticketinfo-service'], 1651130878213, 1651131182441), (['ts-station-service', 'ts-ticketinfo-service'], 1651131373128, 1651131677745), (['ts-ticketinfo-service', 'ts-travel-service'], 1651131868441, 1651132175552), (['ts-route-service', 'ts-order-service'], 1651132366228, 1651132668044), (['ts-route-service', 'ts-user-service'], 1651132858771, 1651133164786), (['ts-order-service', 'ts-station-service'], 1651133355475, 1651133666996), (['ts-ticketinfo-service', 'ts-basic-service'], 1651133857719, 1651134161416), (['ts-order-service', 'ts-route-service'], 1651134352120, 1651134658331), (['ts-basic-service', 'ts-ticketinfo-service'], 1651134849028, 1651135160751), (['ts-travel-service', 'ts-order-service'], 1651135351446, 1651135655550), (['ts-user-service', 'ts-travel-service'], 1651135846241, 1651136196021), (['ts-travel-plan-service', 'ts-route-service'], 1651136459710, 1651136766511), (['ts-travel-service', 'ts-basic-service'], 1651136957194, 1651137260613), (['ts-user-service', 'ts-user-service'], 1651137451297, 1651137753398), (['ts-route-service', 'ts-travel-plan-service'], 1651137944080, 1651138250610), (['ts-travel-service', 'ts-user-service'], 1651138441339, 1651138771912), (['ts-ticketinfo-service', 'ts-station-service'], 1651138962602, 1651139270496), (['ts-basic-service', 'ts-station-service'], 1651139461183, 1651139764278), (['ts-station-service', 'ts-route-service'], 1651139954978, 1651140259892), (['ts-order-service', 'ts-route-service'], 1651140450578, 1651140756479), (['ts-travel-plan-service', 'ts-basic-service'], 1651140947170, 1651141253892), (['ts-basic-service', 'ts-order-service'], 1651141444581, 1651141749686), (['ts-route-service', 'ts-travel-plan-service'], 1651141940390, 1651142245395), (['ts-station-service', 'ts-user-service'], 1651142436079, 1651142741087), (['ts-route-service', 'ts-travel-plan-service'], 1651142931794, 1651143239102)]    
    
    # 1 ab avail
    # request_period_log = [(['ts-route-service'], 1651046006930, 1651046345686), (['ts-travel-service'], 1651046536088, 1651046850818), (['ts-travel-service'], 1651047041209, 1651047358139), (['ts-travel-service'], 1651047548538, 1651047851939), (['ts-station-service'], 1651048042339, 1651048350651), (['ts-basic-service'], 1651048541045, 1651048845560), (['ts-ticketinfo-service'], 1651049035962, 1651049344175), (['ts-user-service'], 1651049534567, 1651049840578), (['ts-order-service'], 1651050030993, 1651050344433), (['ts-order-service'], 1651050534833, 1651050835951), (['ts-order-service'], 1651051026344, 1651051330754), (['ts-basic-service'], 1651051521158, 1651051823554), (['ts-route-service'], 1651052013944, 1651052327763), (['ts-user-service'], 1651052518160, 1651052831373), (['ts-station-service'], 1651053021780, 1651053323273), (['ts-ticketinfo-service'], 1651053513670, 1651053822084), (['ts-travel-plan-service'], 1651054012489, 1651054320797), (['ts-travel-plan-service'], 1651054511205, 1651054811390), (['ts-user-service'], 1651055001769, 1651055315300), (['ts-basic-service'], 1651055505680, 1651055807276), (['ts-ticketinfo-service'], 1651055997683, 1651056300188), (['ts-station-service'], 1651056490588, 1651056792483), (['ts-route-service'], 1651056982884, 1651057293103), (['ts-route-service'], 1651057483499, 1651057794422), (['ts-travel-service'], 1651057984815, 1651058299444), (['ts-route-service'], 1651058489844, 1651058794346), (['ts-basic-service'], 1651058984767, 1651059291785), (['ts-travel-service'], 1651059482197, 1651059794618), (['ts-user-service'], 1651059985046, 1651060286761), (['ts-ticketinfo-service'], 1651060477172, 1651060785597), (['ts-travel-service'], 1651060976006, 1651061279210), (['ts-order-service'], 1651061469606, 1651061775714), (['ts-station-service'], 1651061966116, 1651062273040), (['ts-user-service'], 1651062463443, 1651062766453), (['ts-order-service'], 1651062956853, 1651063269576), (['ts-travel-plan-service'], 1651063459994, 1651063765512), (['ts-order-service'], 1651063955923, 1651064257235), (['ts-basic-service'], 1651064447629, 1651064747737), (['ts-basic-service'], 1651064938128, 1651065239123), (['ts-route-service'], 1651065429530, 1651065739847), (['ts-station-service'], 1651065930265, 1651066234771), (['ts-ticketinfo-service'], 1651066425168, 1651066734296), (['ts-station-service'], 1651066924702, 1651067232717), (['ts-route-service'], 1651067423108, 1651067726422), (['ts-travel-plan-service'], 1651067916830, 1651068230154), (['ts-user-service'], 1651068420557, 1651068729165), (['ts-ticketinfo-service'], 1651068919563, 1651069233102), (['ts-route-service'], 1651069423503, 1651069724297), (['ts-order-service'], 1651069914705, 1651070219820), (['ts-travel-service'], 1651070410223, 1651070723238)]

    # 1 change
    # request_period_log = [(['ts-ticketinfo-service'], 1651336771407, 1651337073614), (['ts-order-service'], 1651337353866, 1651337670661), (['ts-route-service'], 1651337960924, 1651338266292), (['ts-auth-service'], 1651338536520, 1651338838417), (['ts-auth-service'], 1651339717903, 1651340040094), (['ts-ticketinfo-service'], 1651341459289, 1651341763230), (['ts-order-service'], 1651342053491, 1651342353625), (['ts-route-service'], 1651343209699, 1651343632293), (['ts-user-service'], 1651343812415, 1651344118791), (['ts-order-service'], 1651344298911, 1651344626959), (['ts-route-service'], 1651344907217, 1651345313611), (['ts-order-service'], 1651346178704, 1651346488340), (['ts-route-service'], 1651346778607, 1651347108999), (['ts-order-service'], 1651348469297, 1651348774978), (['ts-route-service'], 1651348955096, 1651349267750), (['ts-user-service'], 1651349537977, 1651349841042), (['ts-order-service'], 1651350121292, 1651350425878), (['ts-route-service'], 1651351961208, 1651352265932), (['ts-ticketinfo-service'], 1651354237561, 1651354541431), (['ts-order-service'], 1651355432537, 1651355755860), (['ts-route-service'], 1651355935978, 1651356256056), (['ts-user-service'], 1651356436174, 1651356741150), (['ts-travel-service'], 1651356921268, 1651357224798), (['ts-ticketinfo-service'], 1651357404915, 1651357711162), (['ts-order-service'], 1651357891281, 1651358211460), (['ts-ticketinfo-service'], 1651358391580, 1651358697509), (['ts-order-service'], 1651358977766, 1651359283251), (['ts-order-service'], 1651359463374, 1651359765050), (['ts-order-service'], 1651361188062, 1651361511611), (['ts-route-service'], 1651363000061, 1651363346388), (['ts-order-service'], 1651363636661, 1651363945310)]

    # 1 ab new 5-6
    # request_period_log = [(['ts-seat-service'], 1651748486185, 1651748797147), (['ts-basic-service'], 1651748987478, 1651749295867), (['ts-ticketinfo-service'], 1651749486209, 1651749795299), (['ts-order-other-service'], 1651749985634, 1651750293217), (['ts-consign-service'], 1651750483565, 1651750791138), (['ts-price-service'], 1651750981502, 1651751287378), (['ts-travel-service'], 1651751477713, 1651751902504), (['ts-route-service'], 1651752092854, 1651752394833), (['ts-travel-service'], 1651752585171, 1651752893040), (['ts-train-service'], 1651753083378, 1651753394162), (['ts-config-service'], 1651753584488, 1651753888468), (['ts-order-service'], 1651754078802, 1651754385570), (['ts-route-service'], 1651754575912, 1651754881391), (['ts-train-service'], 1651755071738, 1651755376210), (['ts-user-service'], 1651755566548, 1651755874017), (['ts-route-service'], 1651756064360, 1651756365927), (['ts-ticketinfo-service'], 1651756556284, 1651756859350), (['ts-price-service'], 1651757049709, 1651757354076), (['ts-basic-service'], 1651757544405, 1651757851873), (['ts-user-service'], 1651758042207, 1651758348752), (['ts-order-other-service'], 1651758539096, 1651758846454), (['ts-price-service'], 1651759036801, 1651759343752), (['ts-order-other-service'], 1651759534094, 1651759839642), (['ts-travel-plan-service'], 1651760029972, 1651760337711), (['ts-train-service'], 1651760528062, 
    
    # change new 5-10
    request_period_log = [(['ts-user-service'], 1652082509749, 1652083413427), (['ts-order-service'], 1652083663660, 1652084565731), (['ts-ticketinfo-service'], 1652085969552, 1652086874918), (['ts-user-service'], 1652089451055, 1652090354006), (['ts-order-service'], 1652091758536, 1652092661425), (['ts-route-service'], 1652092841565, 1652093769877), (['ts-travel-service'], 1652094250394, 1652095155270), (['ts-order-service'], 1652095335407, 1652096237776), (['ts-route-service'], 1652096488003, 1652097391167), (['ts-travel-service'], 1652100191897, 1652101095460), (['ts-route-service'], 1652102519903, 1652103423257), (['ts-user-service'], 1652103603394, 1652104537479), (['ts-order-service'], 1652106039521, 1652106995920), (['ts-order-service'], 1652107176058, 1652108144193), (['ts-ticketinfo-service'], 1652108394417, 1652109344989), (['ts-user-service'], 1652109525119, 1652110531415), (['ts-travel-service'], 1652112178207, 1652113127882), (['ts-route-service'], 1652113378101, 1652114336493), (['ts-auth-service'], 1652114586728, 1652115591876), (['ts-route-service'], 1652115772008, 1652116725466), (['ts-user-service'], 1652116975688, 1652117962611), (['ts-auth-service'], 1652119424828, 1652120364654), (['ts-order-service'], 1652122941765, 1652123950378), (['ts-order-service'], 1652127697299, 1652128601830), (['ts-ticketinfo-service'], 1652132342339, 1652133244902), (['ts-route-service'], 1652133495128, 1652134473369), (['ts-route-service'], 1652134723596, 1652135654796), (['ts-auth-service'], 1652135905018, 1652136830032), (['ts-order-service'], 1652137080255, 1652137983164), (['ts-route-service'], 1652138233396, 1652139203956), (['ts-route-service'], 1652139384092, 1652140289239)]

    Two_error = False

    ad_count_list = list()
    for root_cause_item in request_period_log:
        ad_count_list.append(0)

    check_seq = ['start', 'ts-food-service', 'ts-food-map-service', 'ts-travel-service', 'ts-route-service']
    
    # ['start', 'ts-travel2-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-station-service', 'ts-train-service', 'ts-route-service', 'ts-price-service', 'ts-station-service', 'ts-station-service', 'ts-order-other-service']
    # 'fa44402657254e0783df6e549391f2f5.41.16517500408500063'
    # 'fa44402657254e0783df6e549391f2f5.43.16517500346880063'
    # 'fa44402657254e0783df6e549391f2f5.44.16517500316220079'
    # 'fa44402657254e0783df6e549391f2f5.45.16517499861410057' 
    # 'fa44402657254e0783df6e549391f2f5.40.16517499862610069' 
    # 'fa44402657254e0783df6e549391f2f5.44.16517499933750077' 
    # 'fa44402657254e0783df6e549391f2f5.41.16517500006010059' 
    # 'fa44402657254e0783df6e549391f2f5.37.16517499969150079' 
    # 'fa44402657254e0783df6e549391f2f5.41.16517500200070061'
    # 'fa44402657254e0783df6e549391f2f5.39.16517500238830067'
    # 'fa44402657254e0783df6e549391f2f5.44.16517500484670081'

    # ['start', 'ts-travel2-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-station-service', 'ts-train-service', 'ts-route-service', 'ts-price-service', 'ts-station-service', 'ts-station-service', 'ts-order-other-service']
    # 'fa44402657254e0783df6e549391f2f5.45.16517502209030081'
    # 'fa44402657254e0783df6e549391f2f5.37.16517502068740101'
    # 'fa44402657254e0783df6e549391f2f5.46.16517502017140091'
    # 'fa44402657254e0783df6e549391f2f5.40.16517500310800073'
    # 'fa44402657254e0783df6e549391f2f5.46.16517500260180079'

    # ['start', 'ts-preserve-service', 'ts-security-service', 'ts-order-service', 'ts-order-other-service']
    # '61316a79a0c94a16be9070badfe6be0a.40.16517499895550005' 
    # '61316a79a0c94a16be9070badfe6be0a.38.16517499898850001' 
    # '61316a79a0c94a16be9070badfe6be0a.43.16517499972030003'
    # '61316a79a0c94a16be9070badfe6be0a.44.16517500012760001'
    # '61316a79a0c94a16be9070badfe6be0a.43.16517500244250005' 

    # ['start', 'ts-food-service', 'ts-food-map-service', 'ts-travel-service', 'ts-route-service']
    # 'bee80c2085ef4730876fc01fa47eb983.41.16517520933600039'
    # 'bee80c2085ef4730876fc01fa47eb983.47.16517523726890047'
    # 'bee80c2085ef4730876fc01fa47eb983.38.16517523792350035'
    # 'bee80c2085ef4730876fc01fa47eb983.40.16517523819060043'
    
    # ['start', 'ts-travel-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service']
    # '182bb90a7f594af3952b9a74cae6fa8c.42.16517520934500929'
    # '182bb90a7f594af3952b9a74cae6fa8c.45.16517521003900849'
    # '182bb90a7f594af3952b9a74cae6fa8c.39.16517521006840919'
    # '182bb90a7f594af3952b9a74cae6fa8c.43.16517521035580917'
    # '182bb90a7f594af3952b9a74cae6fa8c.47.16517521075120931'
    # '182bb90a7f594af3952b9a74cae6fa8c.40.16517521076950939'
    # '182bb90a7f594af3952b9a74cae6fa8c.41.16517521108760925'
    # '182bb90a7f594af3952b9a74cae6fa8c.45.16517521146500851'
    # '182bb90a7f594af3952b9a74cae6fa8c.44.16517523654610973'

    # ['start', 'ts-travel2-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service']
    # 'fa44402657254e0783df6e549391f2f5.39.16517520961980269'
    # '182bb90a7f594af3952b9a74cae6fa8c.40.16517520964600937'
    # '182bb90a7f594af3952b9a74cae6fa8c.41.16517520964780923'
    # 'fa44402657254e0783df6e549391f2f5.37.16517521005220267'
    # 'fa44402657254e0783df6e549391f2f5.41.16517521037090257'
    # 'fa44402657254e0783df6e549391f2f5.40.16517521107300257'

    # ['start', 'ts-travel-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-station-service', 'ts-train-service', 'ts-route-service', 'ts-price-service', 'ts-station-service', 'ts-station-service', 'ts-order-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-seat-service', 'ts-travel-service', 'ts-route-service']
    # '182bb90a7f594af3952b9a74cae6fa8c.42.16517527203691033'

    # ['start', 'ts-travel-plan-service', 'ts-route-plan-service', 'ts-travel-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-station-service', 'ts-train-service', 'ts-route-service', 'ts-price-service', 'ts-station-service', 'ts-station-service', 'ts-order-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-seat-service', 'ts-travel-service', 'ts-route-service']
    # 'df107144ddb94b1e88946e8898493786.36.16517527205250061'

    # ['start', 'ts-travel-plan-service', 'ts-route-plan-service', 'ts-travel-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-station-service', 'ts-train-service', 'ts-route-service', 'ts-price-service', 'ts-station-service', 'ts-station-service', 'ts-order-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-seat-service', 'ts-travel-service', 'ts-route-service']
    # 'df107144ddb94b1e88946e8898493786.35.16517527280640053'
    # 'df107144ddb94b1e88946e8898493786.43.16517527266270051'
    # 'df107144ddb94b1e88946e8898493786.44.16517527269120055'

    # ['start', 'ts-travel-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-station-service', 'ts-train-service', 'ts-route-service', 'ts-price-service', 'ts-station-service', 'ts-station-service', 'ts-order-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-seat-service', 'ts-travel-service', 'ts-route-service']
    # '182bb90a7f594af3952b9a74cae6fa8c.39.16517527245511023'
    # '182bb90a7f594af3952b9a74cae6fa8c.44.16517527235491043'

    # ['start', 'ts-travel-plan-service', 'ts-route-plan-service', 'ts-station-service', 'ts-station-service', 'ts-route-service', 'ts-travel-service']
    # 'df107144ddb94b1e88946e8898493786.42.16517527272870065'
    # 'df107144ddb94b1e88946e8898493786.41.16517527276360071'
    
    service_seq_set_init = list()
    service_set_init = list()
    # file_init = open(r'/home/kagaya/work/TF-RCA/DataPreprocess/data/preprocessed/trainticket/2022-04-20_17-34-08/data.json', 'r')
    # file_init = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-04_16-55-46/data.json', 'r')
    file_init = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-08_14-03-38/data.json', 'r')    # new
    raw_data_init = json.load(file_init)
    label_error_traces_init = list()
    # print('getting trace data (api and time seq) ... 1')
    for trace_id, trace in sorted(raw_data_init.items(), key = lambda i: i[1]['edges']['0'][0]['startTime']):
        service_seq = ['start']
        spans = []
        for span in trace['edges'].values():
            spans.extend(span)
            for item in span:
                if item['service'] not in service_set_init:
                    service_set_init.append(item['service'])
                if (item['isError'] == True) and (trace['abnormal'] == False) and (trace_id not in label_error_traces_init):
                    label_error_traces_init.append(trace_id)
        spans = sorted(spans, key=lambda span: span['startTime'])
        service_seq.extend([span['service'] for span in spans])
        if service_seq == check_seq:
            time_seq = [span['rawDuration'] for span in spans]
            print("find it !")
        if service_seq not in service_seq_set_init:
            service_seq_set_init.append(service_seq)
    

    service_seq_set_format = list()
    service_set_format = list()
    # file_format = open(r'/home/kagaya/work/TF-RCA/DataPreprocess/data/preprocessed/trainticket/2022-04-19_21-01-30/data.json', 'r')
    # file_format = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-30_10-42-02/data.json', 'r')
    
    # 1 ab avail
    # file_format = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-30_14-39-40/data.json', 'r')

    # 1 change
    # file_format = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-01_13-40-58/data.json', 'r')

    # 2 ab
    # file_format = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-30_19-41-29/data.json', 'r')

    # 1 ab new 5-6
    # file_format = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-06_17-28-43/data.json', 'r')

    # change new 5-10
    file_format = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-11_00-06-32/data.json', 'r')    

    raw_data_format = json.load(file_format)

    # print('getting trace data (api and time seq) ... 1')
    ab_count = 0
    label_error_traces_format = list()
    performance_ab_count = 0
    for trace_id, trace in sorted(raw_data_format.items(), key = lambda i: i[1]['edges']['0'][0]['startTime']):  
        service_seq = ['start']
        spans = []
        for span in trace['edges'].values():
            spans.extend(span)
            for item in span:
                if item['service'] not in service_set_format:
                    service_set_format.append(item['service'])
                if (item['isError'] == True) and (trace['abnormal'] == False) and (trace_id not in label_error_traces_format):
                    label_error_traces_format.append(trace_id)
        spans = sorted(spans, key=lambda span: span['startTime'])
        isError_seq = [span['isError'] for span in spans]
        if (True not in isError_seq) and (trace['abnormal']==1):
            performance_ab_count += 1
        service_seq.extend([span['service'] for span in spans])
        if service_seq == check_seq:
            time_seq = [span['rawDuration'] for span in spans]
            print("find it !")
        ab_count = ab_count + (1 if trace['abnormal'] else 0)
        if trace['abnormal'] == 1:
            for idx, root_cause_item in enumerate(request_period_log):
                if trace['edges']['0'][0]['startTime'] >= root_cause_item[1] and trace['edges']['0'][0]['startTime'] <= root_cause_item[2]:
                    ad_count_list[idx] += 1
        if service_seq not in service_seq_set_format:
            service_seq_set_format.append(service_seq)
        
    # print('getting trace data (api and time seq) ... 1')
    in_rate = dict()
    in_count_normal_dict = dict()
    case_data_num = dict()    # normal/abnormal
    case_pattern = dict()    # normal/abnormal
    case_service_pattern = dict()    # {case1: {service1: count1, service2: count2, service3: count3}, case2: {service1: count1, service2: count2}}
    for root_cause_idx in range(len(request_period_log)):
        raw_data = dict()
        root_cause_check_dict = dict()
        pattern = [[], []]    # normal/abnormal
        # in_rate[str(root_cause_idx)]
        root_cause_check_dict[str(request_period_log[root_cause_idx][0])] = []
        case_data_num[str(root_cause_idx)] = [0, 0]
        case_service_pattern[str(root_cause_idx)] = dict()
        for trace_id, trace in sorted(raw_data_format.items(), key = lambda i: i[1]['edges']['0'][0]['startTime']):
            if trace['edges']['0'][0]['startTime']>=request_period_log[root_cause_idx][1] and trace['edges']['0'][0]['startTime']<=request_period_log[root_cause_idx][2]:
                raw_data[trace_id] = trace
        
        in_count = 0
        abnormal_count = 0
        in_count_normal = 0
        for trace_id, trace in sorted(raw_data.items(), key = lambda i: i[1]['edges']['0'][0]['startTime']):
            service_seq = ['start']
            spans = []
            for span in trace['edges'].values():
                spans.extend(span)
            spans = sorted(spans, key=lambda span: span['startTime'])
            service_seq.extend([span['service'] for span in spans])
            time_seq = [span['rawDuration'] for span in spans]
            time_stamp = trace['edges']['0'][0]['startTime']
            if trace['abnormal']==1:
                for service in set(service_seq):
                    if service == 'start':
                        continue
                    if service not in case_service_pattern[str(root_cause_idx)].keys():
                        case_service_pattern[str(root_cause_idx)][service] = 1
                    else:
                        case_service_pattern[str(root_cause_idx)][service] += 1
                if service_seq not in pattern[1]:
                    pattern[1].append(service_seq)
                case_data_num[str(root_cause_idx)][1] += 1
                abnormal_count += 1
                if (Two_error==True) and (request_period_log[root_cause_idx][0][0] in service_seq or request_period_log[root_cause_idx][0][1] in service_seq):
                    in_label = 'in'
                    in_count += 1
                elif (Two_error==False) and (request_period_log[root_cause_idx][0][0] in service_seq):
                    in_label = 'in'
                    in_count += 1
                else:
                    in_label = 'out'
                root_cause_check_dict[str(request_period_log[root_cause_idx][0])].append({'in_or_out': in_label, 'service_seq': service_seq, 'trace_id': trace_id, 'time_stamp': time_stamp})
            elif trace['abnormal']==0:
                if service_seq not in pattern[0]:
                    pattern[0].append(service_seq)
                case_data_num[str(root_cause_idx)][0] += 1
                if (Two_error==True) and (request_period_log[root_cause_idx][0][0] in service_seq or request_period_log[root_cause_idx][0][1] in service_seq):
                    in_count_normal += 1
                elif (Two_error==False) and (request_period_log[root_cause_idx][0][0] in service_seq):
                    in_count_normal += 1

        in_rate[str(root_cause_idx)] = float(in_count/abnormal_count) if abnormal_count!=0 else 0
        in_count_normal_dict[str(root_cause_idx)] = in_count_normal
        case_pattern[str(root_cause_idx)] = [len(pattern[0]), len(pattern[1])]
    
    # differences = service_seq_set_format.difference(service_seq_set_init)    # 只在正式处理阶段出现的 trace 结构
    differences_seq = list()
    for item in service_seq_set_format:
        if item not in service_seq_set_init:
            differences_seq.append(item)
    
    differences_svc = list()
    for item in service_set_format:
        if item not in service_set_init:
            differences_svc.append(item)
    
    return len(differences_seq), len(differences_svc), ab_count, performance_ab_count, ad_count_list, root_cause_check_dict, in_rate, in_count_normal_dict, case_data_num, case_pattern, label_error_traces_init, label_error_traces_format, case_service_pattern

if __name__ == '__main__':
    diff_count = check_match()
    print("{} structure dismatch !".format(diff_count))