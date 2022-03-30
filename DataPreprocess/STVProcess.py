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

def load_dataset():
    trace_list = list()

    with open(r'G:/workspace/TraceCluster/newData/preprocessed/data.json', 'r') as file_1:
        raw_data = json.load(file_1)
        print('getting trace data (api and time seq) ... 1')
        for trace_id, trace in tqdm(raw_data.items()):
            service_seq = ['start']
            spans = []
            for span in trace['edges'].values():
                spans.extend(span)
            spans = sorted(spans, key=lambda span: span['startTime'])
            service_seq.extend([span['service'] for span in spans])
            time_seq = [span['rawDuration'] for span in spans]
            time_stamp = trace['edges']['0'][0]['startTime']
            trace_list.append({'trace_id': trace_id, 'service_seq': service_seq, 'time_seq': time_seq, 'time_stamp': time_stamp, 'trace_bool': trace['abnormal']})

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
    return trace_list


def process_one_trace(trace, unique_path):
    for i in range(1, len(trace['service_seq'])):
        if '->'.join(trace['service_seq'][:i + 1]) not in unique_path:
            unique_path.append('->'.join(trace['service_seq'][:i + 1]))
    return unique_path


def embedding_to_vector(trace, unique_path):
    length = len(unique_path)
    vector_list = [0 for i in range(length)]
    for i in range(1, len(trace['service_seq'])):
        vector_list[unique_path.index('->'.join(trace['service_seq'][:i + 1]))] = trace['time_seq'][i-1]    
    return vector_list