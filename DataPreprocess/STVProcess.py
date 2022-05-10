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
from SpanProcess import preprocess_span

same_seq_test_list = list()

def load_dataset(start, end, dataLevel, stage, raw_data_total=None):
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
        time_seq = [span['rawDuration'] for span in spans]
        time_stamp = trace['edges']['0'][0]['startTime']
        trace_list.append({'trace_id': trace_id, 'service_seq': service_seq, 'time_seq': time_seq, 'time_stamp': time_stamp, 'trace_bool': trace['abnormal']})
        
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
    vector_list = [0 for i in range(length)]
    for i in range(1, len(trace['service_seq'])):
        vector_list[unique_path['->'.join(trace['service_seq'][:i + 1])][1]] = trace['time_seq'][i-1] 
        # vector_list[unique_path.index('->'.join(trace['service_seq'][:i + 1]))] = trace['time_seq'][i-1]    
    return vector_list


def check_match():
    # request_period_log = [(['ts-ticketinfo-service'], 1650641167424, 1650641474258), (['ts-order-service'], 1650641574378, 1650641888464), (['ts-route-service'], 1650641988585, 1650642301332), (['ts-auth-service'], 1650642401453, 1650642734489), (['ts-auth-service'], 1650643240663, 1650643564678), (['ts-ticketinfo-service'], 1650644475959, 1650644780866), (['ts-order-service'], 1650644880993, 1650645186599), (['ts-route-service'], 1650645688256, 1650646003680), (['ts-user-service'], 1650646103797, 1650646403929), (['ts-order-service'], 1650646504048, 1650646810985), (['ts-route-service'], 1650646911106, 1650647229957), (['ts-order-service'], 1650647732560, 1650648038136), (['ts-route-service'], 1650648138255, 1650648476697), (['ts-order-service'], 1650649385278, 1650649690202), (['ts-route-service'], 1650649790322, 1650650108198), (['ts-user-service'], 1650650208317, 1650650510867), (['ts-order-service'], 1650650610988, 1650650915668), (['ts-route-service'], 1650651886870, 1650652192271), (['ts-ticketinfo-service'], 1650653494728, 1650653797611), (['ts-order-service'], 1650654300889, 1650654605778), (['ts-route-service'], 1650654705891, 1650655022755), (['ts-user-service'], 1650655122865, 1650655424565), (['ts-travel-service'], 1650655524681, 1650655828606), (['ts-ticketinfo-service'], 1650655928755, 1650656236254), (['ts-order-service'], 1650656336374, 1650656641766), (['ts-ticketinfo-service'], 1650656741888, 1650657044303), (['ts-order-service'], 1650657144418, 1650657448876), (['ts-order-service'], 1650657548999, 1650657854457), (['ts-order-service'], 1650658788037, 1650659091412), (['ts-route-service'], 1650659997502, 1650660373783), (['ts-order-service'], 1650660473901, 1650660775220)]
    # request_period_log = [(['ts-station-service', 'ts-travel-plan-service'], 1650799190100, 1650799503849), (['ts-travel-plan-service', 'ts-ticketinfo-service'], 1650799694546, 1650800003505), (['ts-travel-plan-service', 'ts-user-service'], 1650800194225, 1650800499167), (['ts-route-service', 'ts-user-service'], 1650800689886, 1650800990422), (['ts-order-service', 'ts-route-service'], 1650801181181, 1650801487521), (['ts-travel-service', 'ts-ticketinfo-service'], 1650801678227, 1650801982366), (['ts-travel-plan-service', 'ts-station-service'], 1650802295697, 1650802603234), (['ts-station-service', 'ts-route-service'], 1650802793966, 1650803098596), (['ts-user-service', 'ts-order-service'], 1650803289302, 1650803594537), (['ts-order-service', 'ts-basic-service'], 1650803785275, 1650804090999), (['ts-ticketinfo-service', 'ts-travel-plan-service'], 1650804281703, 1650804587126), (['ts-ticketinfo-service', 'ts-travel-service'], 1650804777821, 1650805078830), (['ts-travel-service', 'ts-basic-service'], 1650805269546, 1650805573167), (['ts-route-service', 'ts-order-service'], 1650805763890, 1650806067321), (['ts-route-service', 'ts-ticketinfo-service'], 1650806258018, 1650806563350), (['ts-basic-service', 'ts-route-service'], 1650806754077, 1650807058218), (['ts-ticketinfo-service', 'ts-travel-service'], 1650807248906, 1650807554234), (['ts-basic-service', 'ts-basic-service'], 1650807744931, 1650808051765), (['ts-travel-service', 'ts-order-service'], 1650808242448, 1650808600766), (['ts-user-service', 'ts-station-service'], 1650808791455, 1650809097474), (['ts-basic-service', 'ts-travel-service'], 1650809288169, 1650809591884), (['ts-station-service', 'ts-user-service'], 1650809782583, 1650810083310), (['ts-user-service', 'ts-station-service'], 1650810273996, 1650810580125), (['ts-order-service', 'ts-travel-plan-service'], 1650810770817, 1650811073429), (['ts-ticketinfo-service', 'ts-travel-service'], 1650811264128, 1650811569038), (['ts-user-service', 'ts-ticketinfo-service'], 1650811759737, 1650812063955), (['ts-station-service', 'ts-ticketinfo-service'], 1650812254662, 1650812559676), (['ts-ticketinfo-service', 'ts-travel-service'], 1650812750379, 1650813056492), (['ts-route-service', 'ts-order-service'], 1650813247180, 1650813553395), (['ts-route-service', 'ts-user-service'], 1650813744119, 1650814047621), (['ts-order-service', 'ts-station-service'], 1650814238319, 1650814544734), (['ts-ticketinfo-service', 'ts-basic-service'], 1650814735430, 1650815038854), (['ts-order-service', 'ts-route-service'], 1650815229528, 1650815535653), (['ts-basic-service', 'ts-ticketinfo-service'], 1650815726335, 1650816033550), (['ts-travel-service', 'ts-order-service'], 1650816224229, 1650816528146), (['ts-user-service', 'ts-travel-service'], 1650816718855, 1650817024253), (['ts-travel-plan-service', 'ts-travel-plan-service'], 1650817214950, 1650817517647), (['ts-travel-service', 'ts-basic-service'], 1650817708339, 1650818013343), (['ts-user-service', 'ts-user-service'], 1650818204021, 1650818508128), (['ts-travel-plan-service', 'ts-travel-plan-service'], 1650818698827, 1650818999444), (['ts-travel-service', 'ts-user-service'], 1650819190140, 1650819495239), (['ts-station-service', 'ts-station-service'], 1650819685930, 1650819989940), (['ts-basic-service', 'ts-station-service'], 1650820180622, 1650820482333), (['ts-station-service', 'ts-route-service'], 1650820673033, 1650820977730), (['ts-order-service', 'ts-route-service'], 1650821168406, 1650821472008), (['ts-travel-plan-service', 'ts-basic-service'], 1650821662699, 1650821968898), (['ts-basic-service', 'ts-order-service'], 1650822159599, 1650822462807), (['ts-route-service', 'ts-travel-plan-service'], 1650822653528, 1650822953929), (['ts-station-service', 'ts-user-service'], 1650823144636, 1650823444945), (['ts-travel-plan-service', 'ts-travel-plan-service'], 1650823635651, 1650823938547)]

    # request_period_log = [(['ts-route-service'], 1650254059896, 1650254361387), (['ts-travel-service'], 1650254551823, 1650254853456), (['ts-travel-service'], 1650255043883, 1650255344428), (['ts-travel-service'], 1650255534831, 1650255851210), (['ts-station-service'], 1650256151142, 1650256452270), (['ts-basic-service'], 1650256642674, 1650256944234), (['ts-ticketinfo-service'], 1650257134636, 1650257435240), (['ts-user-service'], 1650257625637, 1650257927939), (['ts-order-service'], 1650258118339, 1650258421608), (['ts-order-service'], 1650258611989, 1650258912546), (['ts-order-service'], 1650259102938, 1650259403142), (['ts-basic-service'], 1650259593530, 1650259896864), (['ts-route-service'], 1650260087260, 1650260388746), (['ts-user-service'], 1650260579154, 1650260881122), (['ts-station-service'], 1650261071509, 1650261371783), (['ts-ticketinfo-service'], 1650261562192, 1650261865276), (['ts-travel-plan-service'], 1650262055678, 1650262358656), (['ts-travel-plan-service'], 1650262549050, 1650262852081), (['ts-user-service'], 1650263042497, 1650263343295), (['ts-basic-service'], 1650263533699, 1650263836890), (['ts-ticketinfo-service'], 1650264027284, 1650264328269), (['ts-station-service'], 1650264518660, 1650264821500), (['ts-travel-plan-service'], 1650265011904, 1650265312147), (['ts-route-service'], 1650265502560, 1650265803476), (['ts-travel-service'], 1650265993869, 1650266305823),
    #                       (['ts-route-service'], 1650266496226, 1650266796479), (['ts-basic-service'], 1650266986888, 1650267287653), (['ts-travel-service'], 1650267478060, 1650267779310), (['ts-user-service'], 1650267969704, 1650268271204), (['ts-ticketinfo-service'], 1650268461603, 1650268763257), (['ts-travel-service'], 1650268953663, 1650269257054), (['ts-order-service'], 1650269447456, 1650269749062), (['ts-station-service'], 1650269939458, 1650270241764), (['ts-user-service'], 1650270432172, 1650270732294), (['ts-order-service'], 1650270922698, 1650271225238), (['ts-travel-plan-service'], 1650271415635, 1650271715726), (['ts-order-service'], 1650271906150, 1650272206462), (['ts-basic-service'], 1650272396851, 1650272699948), (['ts-basic-service'], 1650272890362, 1650273192473), (['ts-route-service'], 1650273382897, 1650273684256), (['ts-station-service'], 1650273874658, 1650274177836), (['ts-ticketinfo-service'], 1650274368244, 1650274668476), (['ts-station-service'], 1650274858860, 1650275159546), (['ts-route-service'], 1650275349948, 1650275652839), (['ts-travel-plan-service'], 1650275843231, 1650276144522), (['ts-user-service'], 1650276334921, 1650276636243), (['ts-ticketinfo-service'], 1650276826642, 1650277128245), (['ts-travel-plan-service'], 1650277318638, 1650277619725), (['ts-order-service'], 1650277810111, 1650278110838), (['ts-travel-service'], 1650278301231, 1650278626536)]
    # request_period_log = [(['ts-route-service'], 1650978142584, 1650978445305), (['ts-travel-service'], 1650978635707, 1650978961865), (['ts-travel-service'], 1650979152262, 1650979460973), (['ts-travel-service'], 1650979651364, 1650980004858), (['ts-station-service'], 1650980195250, 1650980499254), (['ts-basic-service'], 1650980689677, 1650980994204), (['ts-ticketinfo-service'], 1650981184602, 1650981493825), (['ts-user-service'], 1650981684256, 1650981992263), (['ts-order-service'], 1650982182661, 1650982483576), (['ts-order-service'], 1650982673982, 1650982979787), (['ts-order-service'], 1650983170180, 1650983480798), (['ts-basic-service'], 1650983671199, 1650983978223), (['ts-route-service'], 1650984168626, 1650984472640), (['ts-user-service'], 1650984663050, 1650984964054), (['ts-station-service'], 1650985154468, 1650985458274), (['ts-ticketinfo-service'], 1650985648682, 1650985956000), (['ts-travel-plan-service'], 1650986146416, 1650986454634), (['ts-travel-plan-service'], 1650986645044, 1650986947542), (['ts-user-service'], 1650987137950, 1650987441362), (['ts-basic-service'], 1650987631768, 1650987937475), (['ts-ticketinfo-service'], 1650988127884, 1650988427980), (['ts-station-service'], 1650988618388, 1650988920685), (['ts-route-service'], 1650989111093, 1650989411993), (['ts-route-service'], 1650989602376, 1650989903480), (['ts-travel-service'], 1650990093885, 1650990399800),
    #                   (['ts-route-service'], 1650990590187, 1650990890385), (['ts-basic-service'], 1650991080781, 1650991382580), (['ts-travel-service'], 1650991572990, 1650991880294), (['ts-user-service'], 1650992070708, 1650992371601), (['ts-ticketinfo-service'], 1650992561997, 1650992863287), (['ts-travel-service'], 1650993053718, 1650993357412), (['ts-order-service'], 1650993547806, 1650993847907), (['ts-station-service'], 1650994038392, 1650994338497), (['ts-user-service'], 1650994528885, 1650994833185), (['ts-order-service'], 1650995023587, 1650995330885), (['ts-travel-plan-service'], 1650995521277, 1650995822762), (['ts-order-service'], 1650996013157, 1650996318547), (['ts-basic-service'], 1650996508925, 1650996812521), (['ts-basic-service'], 1650997002915, 1650997306422), (['ts-route-service'], 1650997496827, 1650997802820), (['ts-station-service'], 1650997993213, 1650998300522), (['ts-ticketinfo-service'], 1650998490929, 1650998799331), (['ts-station-service'], 1650998989720, 1650999297412), (['ts-route-service'], 1650999487855, 1650999788542), (['ts-travel-plan-service'], 1650999978950, 1651000281637), (['ts-user-service'], 1651000472026, 1651000776508), (['ts-ticketinfo-service'], 1651000966967, 1651001273564), (['ts-route-service'], 1651001463964, 1651001772659), (['ts-order-service'], 1651001963058, 1651002263129), (['ts-travel-service'], 1651002453539, 1651002760436)]
    # request_period_log = [(['ts-ticketinfo-service'], 1651158584541, 1651158886215), (['ts-order-service'], 1651158986338, 1651159288964), (['ts-route-service'], 1651159389082, 1651159722795), (['ts-auth-service'], 1651159822914, 1651160126690), (['ts-auth-service'], 1651160627391, 1651160947340), (['ts-ticketinfo-service'], 1651161848843, 1651162152412), (['ts-order-service'], 1651162252534, 1651162553132), (['ts-route-service'], 1651163055532, 1651163369548), (['ts-user-service'], 1651163469669, 1651163770339), (['ts-order-service'], 1651163870459, 1651164170950), (['ts-route-service'], 1651164271073, 1651164572177), (['ts-order-service'], 1651165074160, 1651165377775), (['ts-route-service'], 1651165477893, 1651165788566), (['ts-order-service'], 1651166694458, 1651166997823), (['ts-route-service'], 1651167097942, 1651167413223), (['ts-user-service'], 1651167513343, 1651167814067), (['ts-order-service'], 1651167914187, 1651168215403), (['ts-route-service'], 1651169119140, 1651169421431), (['ts-ticketinfo-service'], 1651170731284, 1651171033809), (['ts-order-service'], 1651171535176, 1651171836939), (['ts-route-service'], 1651171937058, 1651172252328), (['ts-user-service'], 1651172352448, 1651172653635), (['ts-travel-service'], 1651172753758, 1651173054727), (['ts-ticketinfo-service'], 1651173154874, 1651173456703), (['ts-order-service'], 1651173556824, 1651173859667), (['ts-ticketinfo-service'], 1651173959786, 1651174261111), (['ts-order-service'], 1651174361230, 1651174661242), (['ts-order-service'], 1651174761362, 1651175064931), (['ts-order-service'], 1651175968209, 1651176272453), (['ts-route-service'], 1651177217105, 1651177559554), (['ts-order-service'], 1651177659673, 1651177961553)]
    # request_period_log = [(['ts-ticketinfo-service'], 1651336771407, 1651337073614), (['ts-order-service'], 1651337353866, 1651337670661), (['ts-route-service'], 1651337960924, 1651338266292), (['ts-auth-service'], 1651338536520, 1651338838417), (['ts-auth-service'], 1651339717903, 1651340040094), (['ts-ticketinfo-service'], 1651341459289, 1651341763230), (['ts-order-service'], 1651342053491, 1651342353625), (['ts-route-service'], 1651343209699, 1651343632293), (['ts-user-service'], 1651343812415, 1651344118791), (['ts-order-service'], 1651344298911, 1651344626959), (['ts-route-service'], 1651344907217, 1651345313611), (['ts-order-service'], 1651346178704, 1651346488340), (['ts-route-service'], 1651346778607, 1651347108999), (['ts-order-service'], 1651348469297, 1651348774978), (['ts-route-service'], 1651348955096, 1651349267750), (['ts-user-service'], 1651349537977, 1651349841042), (['ts-order-service'], 1651350121292, 1651350425878), (['ts-route-service'], 1651351961208, 1651352265932), (['ts-ticketinfo-service'], 1651354237561, 1651354541431), (['ts-order-service'], 1651355432537, 1651355755860), (['ts-route-service'], 1651355935978, 1651356256056), (['ts-user-service'], 1651356436174, 1651356741150), (['ts-travel-service'], 1651356921268, 1651357224798), (['ts-ticketinfo-service'], 1651357404915, 1651357711162), (['ts-order-service'], 1651357891281, 1651358211460), (['ts-ticketinfo-service'], 1651358391580, 1651358697509), (['ts-order-service'], 1651358977766, 1651359283251), (['ts-order-service'], 1651359463374, 1651359765050), (['ts-order-service'], 1651361188062, 1651361511611), (['ts-route-service'], 1651363000061, 1651363346388), (['ts-order-service'], 1651363636661, 1651363945310)]
    # request_period_log = [(['ts-route-service'], 1651545511592, 1651545817616), (['ts-travel-service'], 1651546008010, 1651546309329), (['ts-travel-service'], 1651546499736, 1651546799859), (['ts-travel-service'], 1651546990251, 1651547299389), (['ts-station-service'], 1651547489801, 1651547795231), (['ts-basic-service'], 1651547985647, 1651548287376), (['ts-ticketinfo-service'], 1651548477791, 1651548787925), (['ts-user-service'], 1651548978345, 1651549279760), (['ts-order-service'], 1651549470169, 1651549775698), (['ts-order-service'], 1651549966099, 1651550266311), (['ts-order-service'], 1651550456716, 1651550765841), (['ts-basic-service'], 1651550956245, 1651551260875), (['ts-route-service'], 1651551451281, 1651551760015), (['ts-user-service'], 1651551950431, 1651552256166), (['ts-station-service'], 1651552446580, 1651552754805), (['ts-ticketinfo-service'], 1651552945209, 1651553245524), (['ts-travel-plan-service'], 1651553435927, 1651553741073), (['ts-travel-plan-service'], 1651553931462, 1651554235109), (['ts-user-service'], 1651554425526, 1651554732175), (['ts-basic-service'], 1651554922574, 1651555226703), (['ts-ticketinfo-service'], 1651555417107, 1651555717725), (['ts-station-service'], 1651555908153, 1651556216776), (['ts-route-service'], 1651556407180, 1651556714608), (['ts-route-service'], 1651556905010, 1651557206326), (['ts-travel-service'], 1651557396739, 1651557699560),
    #                   (['ts-route-service'], 1651557889965, 1651558195897), (['ts-basic-service'], 1651558386290, 1651558691631), (['ts-travel-service'], 1651558882046, 1651559185174), (['ts-user-service'], 1651559375572, 1651559677708), (['ts-ticketinfo-service'], 1651559868115, 1651560170650), (['ts-travel-service'], 1651560361052, 1651560665378), (['ts-order-service'], 1651560855790, 1651561156413), (['ts-station-service'], 1651561346824, 1651561649544), (['ts-user-service'], 1651561839955, 1651562147284), (['ts-order-service'], 1651562337694, 1651562642626), (['ts-travel-plan-service'], 1651562833054, 1651563140886), (['ts-order-service'], 1651563331281, 1651563636227), (['ts-basic-service'], 1651563826665, 1651564130502), (['ts-basic-service'], 1651564320907, 1651564624737), (['ts-route-service'], 1651564815152, 1651565122901), (['ts-station-service'], 1651565313309, 1651565617837), (['ts-ticketinfo-service'], 1651565808250, 1651566108573), (['ts-station-service'], 1651566298986, 1651566599625), (['ts-route-service'], 1651566790024, 1651567093260), (['ts-travel-plan-service'], 1651567283678, 1651567593419), (['ts-user-service'], 1651567783811, 1651568087445), (['ts-ticketinfo-service'], 1651568277852, 1651568580274), (['ts-route-service'], 1651568770680, 1651569073306), (['ts-order-service'], 1651569263719, 1651569567951), (['ts-travel-service'], 1651569758356, 1651570065489)]
    # request_period_log = [(['ts-seat-service'], 1651651914438, 1651652223386), (['ts-basic-service'], 1651652413788, 1651652718618), (['ts-ticketinfo-service'], 1651652909019, 1651653214154), (['ts-order-other-service'], 1651653404549, 1651653705472), (['ts-consign-service'], 1651653895895, 1651654200739), (['ts-price-service'], 1651654391132, 1651654696069), (['ts-travel-service'], 1651654886494, 1651655249419), (['ts-route-service'], 1651655502082, 1651655805608), (['ts-travel-service'], 1651655996019, 1651656298733), (['ts-train-service'], 1651656489142, 1651656793970), (['ts-config-service'], 1651656984387, 1651657287318), (['ts-order-service'], 1651657477713, 1651657781729), (['ts-route-service'], 1651657972145, 1651658277172), (['ts-train-service'], 1651658467579, 1651658772998), (['ts-user-service'], 1651658963419, 1651659267836), (['ts-route-service'], 1651659458258, 1651659761384), (['ts-ticketinfo-service'], 1651659951782, 1651660256311), (['ts-price-service'], 1651660446723, 1651660751743), (['ts-basic-service'], 1651660942170, 1651661244194), (['ts-user-service'], 1651661434587, 1651661737316), (['ts-order-other-service'], 1651661927738, 1651662231066), (['ts-price-service'], 1651662421466, 1651662723712), (['ts-order-other-service'], 1651662914115, 1651663217854), (['ts-travel-plan-service'], 1651663408262, 1651663711516), (['ts-train-service'], 1651663901912, 1651664207042), (['ts-config-service'], 1651664397443, 1651664702365), (['ts-rebook-service'], 1651664892776, 1651665198006), (['ts-consign-service'], 1651665388412, 1651665692756), (['ts-order-service'], 1651665883164, 1651666188004), (['ts-route-service'], 1651666378413, 1651666682241), (['ts-rebook-service'], 1651666872666, 1651667176891), (['ts-station-service'], 1651667367299, 1651667672037), (['ts-ticketinfo-service'], 1651667862456, 1651668167687), (['ts-travel-service'], 1651668358079, 1651668661205), (['ts-seat-service'], 1651668851601, 1651669156528), (['ts-rebook-service'], 1651669346924, 1651669651854), (['ts-station-service'], 1651669842259, 1651670145786), (['ts-config-service'], 1651670336208, 1651670641926), (['ts-consign-service'], 1651670832327, 1651671136256), (['ts-station-service'], 1651671326655, 1651671631774), (['ts-order-service'], 1651671822191, 1651672126023), (['ts-basic-service'], 1651672316444, 1651672620465), (['ts-travel-plan-service'], 1651672810874, 1651673113294), (['ts-seat-service'], 1651673303720, 1651673606642), (['ts-user-service'], 1651673797042, 1651674102058), (['ts-basic-service'], 1651674292467, 1651674593087), (['ts-basic-service'], 1651674783504, 1651675087413), (['ts-basic-service'], 1651675277828, 1651675580851), (['ts-order-service'], 1651675771242, 1651676075774), (['ts-order-service'], 1651676266188, 1651676571211)]
    # request_period_log = [(['ts-seat-service'], 1651748486185, 1651748797147), (['ts-basic-service'], 1651748987478, 1651749295867), (['ts-ticketinfo-service'], 1651749486209, 1651749795299), (['ts-order-other-service'], 1651749985634, 1651750293217), (['ts-consign-service'], 1651750483565, 1651750791138), (['ts-price-service'], 1651750981502, 1651751287378), (['ts-travel-service'], 1651751477713, 1651751902504), (['ts-route-service'], 1651752092854, 1651752394833), (['ts-travel-service'], 1651752585171, 1651752893040), (['ts-train-service'], 1651753083378, 1651753394162), (['ts-config-service'], 1651753584488, 1651753888468), (['ts-order-service'], 1651754078802, 1651754385570), (['ts-route-service'], 1651754575912, 1651754881391), (['ts-train-service'], 1651755071738, 1651755376210), (['ts-user-service'], 1651755566548, 1651755874017), (['ts-route-service'], 1651756064360, 1651756365927), (['ts-ticketinfo-service'], 1651756556284, 1651756859350), (['ts-price-service'], 1651757049709, 1651757354076), (['ts-basic-service'], 1651757544405, 1651757851873), (['ts-user-service'], 1651758042207, 1651758348752), (['ts-order-other-service'], 1651758539096, 1651758846454), (['ts-price-service'], 1651759036801, 1651759343752), (['ts-order-other-service'], 1651759534094, 1651759839642), (['ts-travel-plan-service'], 1651760029972, 1651760337711), (['ts-train-service'], 1651760528062,
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        1651760834111), (['ts-config-service'], 1651761024450, 1651761331805), (['ts-rebook-service'], 1651761522155, 1651761825288), (['ts-consign-service'], 1651762015613, 1651762323166), (['ts-order-service'], 1651762513510, 1651762820468), (['ts-route-service'], 1651763010837, 1651763316689), (['ts-rebook-service'], 1651763507033, 1651763866086), (['ts-station-service'], 1651764056441, 1651764363577), (['ts-ticketinfo-service'], 1651764553920, 1651764859280), (['ts-travel-service'], 1651765049611, 1651765355247), (['ts-seat-service'], 1651765545589, 1651765851820), (['ts-rebook-service'], 1651766042163, 1651766349694), (['ts-station-service'], 1651766540051, 1651766845284), (['ts-config-service'], 1651767035612, 1651767337932), (['ts-consign-service'], 1651767528273, 1651767835018), (['ts-station-service'], 1651768025355, 1651768329778), (['ts-order-service'], 1651768520117, 1651768825535), (['ts-basic-service'], 1651769015884, 1651769317814), (['ts-travel-plan-service'], 1651769508162, 1651769810980), (['ts-seat-service'], 1651770001315, 1651770307832), (['ts-user-service'], 1651770498181, 1651770805499), (['ts-basic-service'], 1651770995821, 1651771303663), (['ts-basic-service'], 1651771494025, 1651771796434), (['ts-basic-service'], 1651771986783, 1651772289847), (['ts-order-service'], 1651772480173, 1651772786498), (['ts-order-service'], 1651772976840, 1651773281769)]

    request_period_log = [(['ts-seat-service'], 1651909275478, 1651909578676), (['ts-basic-service'], 1651909769026, 1651910072809), (['ts-ticketinfo-service'], 1651910263159, 1651910567565), (['ts-order-other-service'], 1651910757906, 1651911060704), (['ts-consign-service'], 1651911251059, 1651911556671), (['ts-price-service'], 1651911747025, 1651912049840), (['ts-travel-service'], 1651912240181, 1651912667164), (['ts-route-service'], 1651912857512, 1651913161218), (['ts-travel-service'], 1651913351574, 1651913654665), (['ts-train-service'], 1651913845004, 1651914148293), (['ts-config-service'], 1651914338643, 1651914641538), (['ts-order-service'], 1651914831896, 1651915135077), (['ts-route-service'], 1651915325443, 1651915628635), (['ts-train-service'], 1651915818975, 1651916122358), (['ts-user-service'], 1651916312708, 1651916615998), (['ts-route-service'], 1651916806340, 1651917109523), (['ts-ticketinfo-service'], 1651917299864, 1651917601843), (['ts-price-service'], 1651917792204, 1651918093794), (['ts-basic-service'], 1651918284141, 1651918587744), (['ts-user-service'], 1651918778089, 1651919082980), (['ts-order-other-service'], 1651919273329, 1651919576727), (['ts-price-service'], 1651919767082, 1651920070287), (['ts-order-other-service'], 1651920260639, 1651920564326), (['ts-travel-plan-service'], 1651920754663, 1651921056654), (['ts-train-service'], 1651921247026,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           1651921562348), (['ts-config-service'], 1651921752706, 1651922060621), (['ts-rebook-service'], 1651922250968, 1651922557660), (['ts-consign-service'], 1651922748020, 1651923052603), (['ts-order-service'], 1651923242951, 1651923548853), (['ts-route-service'], 1651923739203, 1651924041695), (['ts-rebook-service'], 1651924232040, 1651924545352), (['ts-station-service'], 1651924735727, 1651925043543), (['ts-ticketinfo-service'], 1651925233888, 1651925551309), (['ts-travel-service'], 1651925741664, 1651926047068), (['ts-seat-service'], 1651926237403, 1651926540815), (['ts-rebook-service'], 1651926731162, 1651927034261), (['ts-station-service'], 1651927224603, 1651927530305), (['ts-config-service'], 1651927720668, 1651928022561), (['ts-consign-service'], 1651928212915, 1651928517607), (['ts-station-service'], 1651928707945, 1651929012525), (['ts-order-service'], 1651929202871, 1651929520297), (['ts-basic-service'], 1651929710660, 1651930013442), (['ts-travel-plan-service'], 1651930203790, 1651930506979), (['ts-seat-service'], 1651930697320, 1651931005107), (['ts-user-service'], 1651931195457, 1651931504044), (['ts-basic-service'], 1651931694382, 1651931998368), (['ts-basic-service'], 1651932188725, 1651932491383), (['ts-basic-service'], 1651932681733, 1651932983017), (['ts-order-service'], 1651933173379, 1651933480263), (['ts-order-service'], 1651933670601, 1651933984586)]
    ad_count_list = list()
    for root_cause_item in request_period_log:
        ad_count_list.append(0)


    service_seq_set_init = list()
    file_init = open(r'/home/kagaya/work/TF-RCA/DataPreprocess/data/preprocessed/trainticket/2022-04-20_17-34-08/data.json', 'r')
    raw_data_init = json.load(file_init)

    # print('getting trace data (api and time seq) ... 1')
    for trace_id, trace in sorted(raw_data_init.items(), key = lambda i: i[1]['edges']['0'][0]['startTime']):
        service_seq = ['start']
        spans = []
        for span in trace['edges'].values():
            spans.extend(span)
        spans = sorted(spans, key=lambda span: span['startTime'])
        service_seq.extend([span['service'] for span in spans])
        if service_seq not in service_seq_set_init:
            service_seq_set_init.append(service_seq)
    

    service_seq_set_format = list()
    # file_format = open(r'/home/kagaya/work/TF-RCA/DataPreprocess/data/preprocessed/trainticket/2022-04-19_21-01-30/data.json', 'r')
    # file_format = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-05_16-38-16/data.json', 'r')
    file_format = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-09_17-12-40/data.json', 'r')

    raw_data_format = json.load(file_format)

    # print('getting trace data (api and time seq) ... 1')
    ab_count = 0
    for trace_id, trace in sorted(raw_data_format.items(), key = lambda i: i[1]['edges']['0'][0]['startTime']):
        service_seq = ['start']
        spans = []
        for span in trace['edges'].values():
            spans.extend(span)
        spans = sorted(spans, key=lambda span: span['startTime'])
        service_seq.extend([span['service'] for span in spans])
        ab_count = ab_count + (1 if trace['abnormal'] else 0)
        if trace['abnormal'] == 1:
            for idx, root_cause_item in enumerate(request_period_log):
                if trace['edges']['0'][0]['startTime'] >= root_cause_item[1] and trace['edges']['0'][0]['startTime'] < root_cause_item[2]:
                    ad_count_list[idx] += 1
        if service_seq not in service_seq_set_format:
            service_seq_set_format.append(service_seq)

    # differences = service_seq_set_format.difference(service_seq_set_init)    # 只在正式处理阶段出现的 trace 结构
    differences = list()
    for item in service_seq_set_format:
        if item not in service_seq_set_init:
            differences.append(item)
    
    return len(differences), ab_count, ad_count_list

if __name__ == '__main__':
    diff_count = check_match()
    print("{} structure dismatch !".format(diff_count))