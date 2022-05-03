# Kagaya kagaya85@outlook.com
import json
from tkinter.messagebox import NO
from xmlrpc.client import Boolean
import yaml
import os
import sys
import time
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import argparse
from tqdm import tqdm
import utils
from typing import List, Callable, Dict, Tuple
from multiprocessing import cpu_count, Manager, current_process
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import wordninja
from transformers import AutoTokenizer, AutoModel
from enum import Enum
import re

from params import request_period_log
from params import data_path_list, init_data_path_list, mm_data_path_list, init_mm_data_path_list, mm_trace_root_list, aiops_data_list


class DataType(Enum):
    TrainTicket = 1
    Wechat = 2
    AIops = 3


data_root = '/data/TraceCluster/raw'
dtype = DataType.TrainTicket
# dtype = DataType.AIops
time_now_str = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
rm_non_rc_abnormal = False

# wecath data flag
use_request = False
cache_file = '/home/kagaya/work/TraceCluster/secrets/cache.json'
embedding_name = ''

# get features of the edge directed to current span
operation_select_keys = ['childrenSpanNum', 'requestDuration', 'responseDuration',
                         'requestAndResponseDuration', 'workDuration', 'subspanNum',
                         'duration', 'rawDuration', 'timeScale']


def normalize(x: float) -> float: return x


def embedding(input: str) -> List[float]:
    return []


# load name cache
cache = {}
mmapis = {}
mm_root_map = {}
service_url = ""
operation_url = ""
sn = ""


class Item:
    def __init__(self) -> None:
        self.SPAN_ID = 'SpanId'
        self.PARENT_SPAN_ID = 'ParentSpan'
        self.TRACE_ID = 'TraceId'
        self.START_TIME = 'StartTime'
        self.END_TIME = 'EndTime'
        self.OPERATION = 'URL'
        self.DURATION = 'Duration'
        self.SPAN_TYPE = 'SpanType'
        self.SERVICE = 'Service'
        self.IS_ERROR = 'IsError'
        self.PEER = 'Peer'
        self.CODE = 'Code'


ITEM = Item()


class Span:
    def __init__(self, raw_span: dict = None) -> None:
        """
        convert raw span to span object
        """
        self.spanId = ''
        self.parentSpanId = ''
        self.traceId = ''
        self.spanType = ''
        self.startTime = 0
        self.duration = 0
        self.service = ''
        self.peer = ''
        self.operation = ''
        self.code = '0'
        self.isError = False

        if raw_span is not None:
            self.spanId = raw_span[ITEM.SPAN_ID]
            self.parentSpanId = raw_span[ITEM.PARENT_SPAN_ID]
            self.traceId = raw_span[ITEM.TRACE_ID]
            self.spanType = raw_span[ITEM.SPAN_TYPE]
            self.startTime = raw_span[ITEM.START_TIME]
            self.duration = raw_span[ITEM.DURATION]
            self.service = str(raw_span[ITEM.SERVICE])
            self.peer = str(raw_span[ITEM.PEER])
            self.operation = str(raw_span[ITEM.OPERATION])
            if ITEM.IS_ERROR in raw_span.keys():
                self.code = str(utils.boolStr2Int(raw_span[ITEM.IS_ERROR]))
                self.isError = utils.any2bool(raw_span[ITEM.IS_ERROR])
            if ITEM.CODE in raw_span.keys():
                self.code = str(raw_span[ITEM.CODE])


def arguments():
    parser = argparse.ArgumentParser(description="Preporcess Argumentes.")
    parser.add_argument('--cores', dest='cores',
                        help='parallel processing core numberes', default=cpu_count())
    parser.add_argument('--wechat', help='use wechat data',
                        action='store_true')
    parser.add_argument('--aiops', help='use aiops data', action='store_true')
    parser.add_argument('--use-request', dest='use_request', help='use http request when replace id to name',
                        action='store_true')
    parser.add_argument('--max-num', dest='max_num',
                        default=100000, help='max trace number in saved file')
    parser.add_argument('--rm-non-rc-abnormal',
                        dest='rm_non_rc_abnormal', action='store_true')
    return parser.parse_args()


def load_mm_span(clickstream_list: List[str], callgraph_list: List[str]) -> Tuple[dict, List[DataFrame]]:
    # wechat data
    raw_spans = []
    root_map = {}

    global cache, mmapis, service_url, operation_url, sn
    if not cache:
        cache = load_name_cache()
    if use_request:
        mmapis = get_mmapi()
        service_url = mmapis['api']['getApps']
        operation_url = mmapis['api']['getModuleInterface']
        sn = mmapis['sn']

    # load root info
    for path in clickstream_list:
        path = os.path.join(data_root, 'wechat', path)
        print(f"loading wechat clickstrem data from {path}")
        clickstreams = pd.read_csv(path)
        for _, root in clickstreams.iterrows():
            root_map[root['GraphIdBase64']] = {
                'spanid': str(root['CallerNodeID']) + str(root['CallerOssID']) + str(root['CallerCmdID']),
                'ossid': str(root['CallerOssID']),
                'cmdid': str(root['CallerCmdID']),
                'nodeid': str(root['CallerNodeID']),
                'code': root['RetCode'],
                'start_time': root['TimeStamp'],
                'cost_time': root['CostTime'],
            }

    # load trace info
    for filepath in callgraph_list:
        filepath = os.path.join(data_root, 'wechat', filepath)
        print(f"loading wechat span data from {filepath}")
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                raw_data = json.load(f)
            mmspans = raw_data['data']
        elif filepath.endswith('.csv'):
            raw_data = pd.read_csv(filepath).drop_duplicates()
            mmspans = raw_data.iterrows()
        else:
            print(f'invalid file type: {filepath}')
            continue

        spans = {
            ITEM.SPAN_ID: [],
            ITEM.PARENT_SPAN_ID: [],
            ITEM.TRACE_ID: [],
            ITEM.SPAN_TYPE: [],
            ITEM.START_TIME: [],
            ITEM.DURATION: [],
            ITEM.SERVICE: [],
            ITEM.PEER: [],
            ITEM.OPERATION: [],
            ITEM.IS_ERROR: [],
            ITEM.CODE: [],
        }

        # convert to dataframe
        for i, s in tqdm(mmspans):
            spans[ITEM.SPAN_ID].append(
                str(s['CalleeNodeID']) + str(s['CalleeCmdID']))
            spans[ITEM.PARENT_SPAN_ID].append(
                str(s['CallerNodeID']) + str(s['CallerCmdID']))
            spans[ITEM.TRACE_ID].append(s['GraphIdBase64'])
            spans[ITEM.SPAN_TYPE].append('Entry')
            spans[ITEM.START_TIME].append(
                int(time.mktime(time.strptime(s['TimeStamp'], "%Y-%m-%d %H:%M:%S"))) * 1000)
            spans[ITEM.DURATION].append(int(s['CostTime']))

            # 尝试替换id为name
            service_name = get_service_name(s['CalleeOssID'])
            if service_name == "":
                spans[ITEM.SERVICE].append(str(s['CalleeOssID']))
            else:
                spans[ITEM.SERVICE].append(service_name)

            spans[ITEM.OPERATION].append(
                get_operation_name(s['CalleeCmdID'], service_name))

            peer_service_name = get_service_name(s['CallerOssID'])
            peer_cmd_name = get_operation_name(
                s['CallerCmdID'], peer_service_name)

            if peer_service_name == "":
                spans[ITEM.PEER].append(
                    '/'.join([str(s['CallerOssID']), peer_cmd_name]))
            else:
                spans[ITEM.PEER].append(
                    '/'.join([peer_service_name, peer_cmd_name]))

            error_code = s['NetworkRet'] if s['NetworkRet'] != 0 else s['ServiceRet']
            spans[ITEM.IS_ERROR].append(utils.int2Bool(error_code))
            spans[ITEM.CODE].append(str(error_code))

        df = DataFrame(spans)
        raw_spans.extend(data_partition(df, 10000))

        return root_map, raw_spans


def load_sw_span(data_path_list: List[str]) -> List[DataFrame]:
    raw_spans = []
    # skywalking data
    for filepath in data_path_list:
        filepath = os.path.join(data_root, 'trainticket', filepath)
        print(f"loading skywalking span data from {filepath}")

        data_type = {ITEM.START_TIME: np.uint64, ITEM.END_TIME: np.uint64}
        spans = pd.read_csv(
            filepath, dtype=data_type
        ).drop_duplicates().dropna()
        spans[ITEM.DURATION] = spans[ITEM.END_TIME] - \
            spans[ITEM.START_TIME]

        spans[ITEM.SERVICE] = spans[ITEM.SERVICE].map(
            lambda x: remove_tail_id(x))
        raw_spans.extend(data_partition(spans, 10000))

    return raw_spans


def load_aiops_span(data_path_list: List[str]) -> List[DataFrame]:
    raw_spans = []

    # load trace info
    for filepath in data_path_list:
        filepath = os.path.join(data_root, 'aiops', filepath)
        print(f"loading aiops span data from {filepath}")
        data_type = {'startTime': np.uint64, 'elapsedTime': np.uint64}
        raw_data = pd.read_csv(filepath, dtype=data_type).drop_duplicates()

        spans = {
            ITEM.SPAN_ID: [],
            ITEM.PARENT_SPAN_ID: [],
            ITEM.TRACE_ID: [],
            ITEM.SPAN_TYPE: [],
            ITEM.START_TIME: [],
            ITEM.DURATION: [],
            ITEM.SERVICE: [],
            ITEM.PEER: [],
            ITEM.OPERATION: [],
            ITEM.IS_ERROR: [],
            ITEM.CODE: [],
        }

        # convert to dataframe
        for _, s in tqdm(raw_data.iterrows()):
            spans[ITEM.SPAN_ID].append(str(s['id']))
            spans[ITEM.PARENT_SPAN_ID].append(str(s['pid']))
            spans[ITEM.TRACE_ID].append(s['traceId'])
            spans[ITEM.SPAN_TYPE].append('Entry')
            spans[ITEM.START_TIME].append(int(s['startTime']))
            spans[ITEM.DURATION].append(int(s['elapsedTime']))
            spans[ITEM.SERVICE].append(
                s['serviceName'] if 'serviceName' in s.keys() else s['dsName'])
            spans[ITEM.OPERATION].append(s['cmdb_id'])
            spans[ITEM.PEER].append('')
            spans[ITEM.IS_ERROR].append(utils.any2bool(s['success']))
            spans[ITEM.CODE].append('')

        df = DataFrame(spans)
        raw_spans.extend(df)

    span_data = pd.concat(raw_spans, axis=0, ignore_index=True)
    return data_partition(span_data, 10000)


def load_span(dtype: DataType, stage: str = 'main') -> List[DataFrame]:
    """
    load raw sapn data from pathList
    """
    raw_spans = []

    if dtype == DataType.Wechat:
        global mm_root_map
        mm_root_map, raw_spans = load_mm_span(
            mm_trace_root_list, mm_data_path_list)
    elif dtype == DataType.TrainTicket:
        raw_spans = load_sw_span(
            data_path_list if stage == 'main' else init_data_path_list)
    elif dtype == DataType.AIops:
        raw_spans = load_aiops_span(aiops_data_list)

    return raw_spans


def remove_tail_id(s: str) -> str:
    x = 'service'
    idx = s.find(x)
    if idx > 0:
        return s[:idx+len(x)]
    return s


def data_partition(data: DataFrame, size: int = 1024) -> List[DataFrame]:
    id_list = data[ITEM.TRACE_ID].unique()
    if len(id_list) < size:
        return [data]

    res = []
    for sub in [id_list[i:i + size] for i in range(0, len(id_list), size)]:
        df = data[data[ITEM.TRACE_ID].isin(sub)]
        res.append(df)

    return res


def build_graph(trace: List[Span], time_normolize: Callable[[float], float], operation_map: dict):
    """
    build trace graph from span list
    """

    trace.sort(key=lambda s: s.startTime)

    if dtype == DataType.Wechat:
        return build_mm_graph(trace, time_normolize, operation_map)
    elif dtype == DataType.TrainTicket:
        return build_sw_graph(trace, time_normolize, operation_map)
    elif dtype == DataType.AIops:
        return build_aiops_graph(trace, time_normolize, operation_map)


def subspan_info(span: Span, child_spans: List[Span]):
    """
    returns subspan duration, subspan number, is_parallel (0-not parallel, 1-is parallel)
    """
    if len(child_spans) == 0:
        return 0, 0, 0
    total_duration = 0
    is_parallel = 0
    time_spans = []
    for child in child_spans:
        time_spans.append(
            {"start": child.startTime, "end": child.startTime + child.duration})
    time_spans.sort(key=lambda s: s["start"])
    last_time_span = time_spans[0]
    last_length = -1

    while (len(time_spans) != last_length):
        last_length = len(time_spans)
        for time_span in time_spans:
            if time_span["start"] < last_time_span["end"]:
                if time_span != time_spans[0]:
                    is_parallel = 1
                    time_span["start"] = last_time_span["start"]
                    time_span["end"] = max(
                        time_span["end"], last_time_span["end"])
                    time_spans.remove(last_time_span)
            last_time_span = time_span
    subspanNum = len(time_spans) + 1

    for time_span in time_spans:
        total_duration += time_span["end"] - time_span["start"]
    if time_spans[0]["start"] == span.startTime:
        subspanNum -= 1
    if time_spans[-1]["end"] == span.startTime + span.duration:
        subspanNum -= 1

    return total_duration, subspanNum, is_parallel


def calculate_edge_features(current_span: Span, trace_duration: dict, spanChildrenMap: dict):
    # base features
    features = {
        'spanId': current_span.spanId,
        'parentSpanId': current_span.parentSpanId,
        'startTime': current_span.startTime,
        'rawDuration': current_span.duration,
        'service': current_span.service,
        'operation': current_span.operation,
        'peer': current_span.peer,
        'isError': current_span.isError,

        'childrenSpanNum': 0,
        'requestDuration': 0,
        'responseDuration': 0,
        'workDuration': 0,
        'timeScale': round(
            (current_span.duration / (trace_duration["end"] - trace_duration["start"])), 4),
        'subspanNum': 0,
        'requestAndResponseDuration': 0,
        'isParallel': 0,
        'callType': 0 if current_span.spanType == "Entry" else 1,
        'statusCode': current_span.code,
    }

    if spanChildrenMap.get(current_span.spanId) is None:
        return features

    children_span = spanChildrenMap[current_span.spanId]
    request_and_response_duration = 0.0
    request_duration = 0.0
    response_duration = 0.0
    children_duration = 0.0
    subspan_duration = 0.0
    subspan_num = 0.0
    min_time = sys.maxsize - 1
    max_time = -1

    for child in children_span:
        if child.startTime < min_time:
            min_time = child.startTime
        if child.startTime + child.duration > max_time:
            max_time = child.startTime + child.duration

        if spanChildrenMap.get(child.spanId) is None:
            continue

        grandChild = spanChildrenMap[child.spanId][0]

        children_duration += grandChild.duration
        child_request_duration = grandChild.startTime - child.startTime

        if child.spanType == "Exit":
            request_duration += child_request_duration
            response_duration += (child.duration -
                                  child_request_duration - grandChild.duration)
            request_and_response_duration += (child.duration -
                                              grandChild.duration)
        elif child.spanType == "Producer":
            if grandChild.startTime + grandChild.duration > trace_duration["end"]:
                trace_duration["end"] = grandChild.startTime + \
                    grandChild.duration

    subspan_duration, subspan_num, is_parallel = subspan_info(
        current_span, children_span)

    # udpate features
    features["isParallel"] = is_parallel
    features["childrenSpanNum"] = len(children_span)
    features["requestDuration"] = request_duration
    features["responseDuration"] = response_duration
    features["requestAndResponseDuration"] = request_and_response_duration
    features["workDuration"] = current_span.duration - subspan_duration
    features["subspanNum"] = subspan_num

    return features


def check_abnormal_span(span: Span) -> str:
    chaos = []
    for set in request_period_log:
        r_start = int(set[1])
        r_end = int(set[2])
        if r_start < span.startTime and span.startTime < r_end:
            chaos.extend(set[0])
            break

    if len(chaos) == 0:
        return ''

    if span.duration < 5000 and not span.isError:
        return ''

    for c in chaos:
        if c == span.service or c == span.peer:
            return c

    return ''


def build_sw_graph(trace: List[Span], time_normolize: Callable[[float], float], operation_map: dict):
    vertexs = {0: ['start', 'start']}
    edges = {}
    trace_duration = {}

    spanIdMap = {'-1': 0}
    spanIdCounter = 1
    rootSpan = None
    spanMap = {}
    spanChildrenMap = {}

    # generate span dict
    has_root = False
    for span in trace:
        if span.parentSpanId == '-1':
            trace_duration["start"] = span.startTime
            trace_duration["end"] = span.startTime + \
                span.duration + 1 if span.duration <= 0 else 0
            has_root = True
        spanMap[span.spanId] = span
        if span.parentSpanId not in spanChildrenMap.keys():
            spanChildrenMap[span.parentSpanId] = []
        spanChildrenMap[span.parentSpanId].append(span)

    if not has_root:
        return None

    # remove local span
    for span in trace:
        if span.spanType != 'Local':
            continue

        if spanMap.get(span.parentSpanId) is None:
            return None
        else:
            local_span_children = spanChildrenMap[span.spanId]
            local_span_parent = spanMap[span.parentSpanId]
            spanChildrenMap[local_span_parent.spanId].remove(span)
            for child in local_span_children:
                child.parentSpanId = local_span_parent.spanId
                spanChildrenMap[local_span_parent.spanId].append(child)

    is_abnormal = 0
    chaos_root = []
    # process other span
    for span in trace:
        """
        (graph object contains Vertexs and Edges
        Edge: [(from, to, duration), ...]
        Vertex: [(id, nodestr), ...]
        """

        # skip client span
        if span.spanType in ['Exit', 'Producer', 'Local']:
            continue

        if span.isError:
            is_abnormal = 1

        # if check_abnormal_span(span):
        root_chaos = check_abnormal_span(span)
        if root_chaos != '':
            chaos_root = [root_chaos]

        # get the parent server span id
        if span.parentSpanId == '-1':
            rootSpan = span
            parentSpanId = '-1'
        else:
            if spanMap.get(span.parentSpanId) is None:
                return None
            parentSpanId = spanMap[span.parentSpanId].parentSpanId

        if parentSpanId not in spanIdMap.keys():
            spanIdMap[parentSpanId] = spanIdCounter
            spanIdCounter += 1

        if span.spanId not in spanIdMap.keys():
            spanIdMap[span.spanId] = spanIdCounter
            spanIdCounter += 1

        vid, pvid = spanIdMap[span.spanId], spanIdMap[parentSpanId]

        # span id should be unique
        if vid not in vertexs.keys():
            ops = span.operation.split('/')
            for i in range(len(ops)):
                if re.match('\{.*\}', ops[i]) != None:
                    ops[i] = '{}'
            opname = '/'.join(ops)
            vertexs[vid] = [span.service, opname]
            span.operation = opname

        if str(pvid) not in edges.keys():
            edges[str(pvid)] = []

        feats = calculate_edge_features(
            span, trace_duration, spanChildrenMap)
        feats['vertexId'] = vid
        feats['duration'] = time_normolize(span.duration)

        edges[str(pvid)].append(feats)

    if rootSpan == None:
        return None

    if is_abnormal == 1 and len(chaos_root) > 0:
        has = False
        for v in vertexs.values():
            if v[0] == chaos_root[0]:
                has = True
                break
        if not has:
            for span in trace:
                if span.spanType == 'Exit' and span.peer == chaos_root[0]:
                    span.service = span.peer
                    ops = span.operation.split('/')

                    if span.service == 'ts-order-service' and ops[4] == 'order':
                        ops[5] = '{}'
                    if span.service == 'ts-food-service' and ops[4] == 'foods':
                        ops[5] = '{}'
                        ops[6] = '{}'
                        ops[7] = '{}'
                    ops[-1] = '{}'
                    if ops[0] == '':
                        ops[0] = 'GET:'

                    opname = '/'.join(ops)
                    vertexs[spanIdCounter] = [span.service, opname]
                    span.operation = opname
                    feats = calculate_edge_features(
                        span, trace_duration, spanChildrenMap)
                    feats['vertexId'] = spanIdCounter
                    feats['duration'] = time_normolize(span.duration)
                    pvid = spanIdMap[span.parentSpanId]
                    if pvid not in edges.keys():
                        edges[str(pvid)] = [feats]
                    else:
                        edges[str(pvid)].append(feats)
                    has = True
                    break
        if not has:
            return None

    if rm_non_rc_abnormal and is_abnormal == 1 and len(chaos_root) == 0:
        return None

    graph = {
        'abnormal': is_abnormal,
        'rc': chaos_root,
        'vertexs': vertexs,
        'edges': edges,
    }

    return graph


def build_aiops_graph(trace: List[Span], time_normolize: Callable[[float], float], operation_map: dict):
    vertexs = {0: ['start', 'start']}
    edges = {}
    trace_duration = {}
    spanIdMap = {'-1': 0}
    spanIdCounter = 1
    rootSpan = None
    spanMap = {}
    spanChildrenMap = {}

    # generate span dict
    for span in trace:
        if span.parentSpanId == 'None':
            span.parentSpanId = '-1'
        spanMap[span.spanId] = span
        if span.parentSpanId not in spanChildrenMap.keys():
            spanChildrenMap[span.parentSpanId] = []
        spanChildrenMap[span.parentSpanId].append(span)

    is_abnormal = 0
    chaos_root = []
    # process other span
    for span in trace:
        """
        (graph object contains Vertexs and Edges
        Edge: [(from, to, duration), ...]
        Vertex: [(id, nodestr), ...]
        """

        # get the parent server span id
        if span.parentSpanId == '-1':
            trace_duration["start"] = span.startTime
            trace_duration["end"] = span.startTime + \
                span.duration + 1 if span.duration <= 0 else 0
            parentSpanId = '-1'
        else:
            parentSpanId = spanMap[span.spanId].parentSpanId

        if parentSpanId not in spanIdMap.keys():
            spanIdMap[parentSpanId] = spanIdCounter
            spanIdCounter += 1

        if span.spanId not in spanIdMap.keys():
            spanIdMap[span.spanId] = spanIdCounter
            spanIdCounter += 1

        vid, pvid = spanIdMap[span.spanId], spanIdMap[parentSpanId]

        # span id should be unique
        if vid not in vertexs.keys():
            vertexs[vid] = [span.service, span.operation]

        if str(pvid) not in edges.keys():
            edges[str(pvid)] = []

        feats = calculate_edge_features(
            span, trace_duration, spanChildrenMap)
        feats['vertexId'] = vid
        feats['duration'] = time_normolize(span.duration)

        edges[str(pvid)].append(feats)

    graph = {
        'abnormal': is_abnormal,
        'rc': chaos_root,
        'vertexs': vertexs,
        'edges': edges,
    }

    return graph


def build_mm_graph(trace: List[Span], time_normolize: Callable[[float], float], operation_map: dict):
    traceId = trace[0].traceId

    spanIdMap = {'-1': 0}
    spanIdCounter = 1
    vertexs = {0: ['start', 'start']}
    edges = {}
    spanMap = {}
    trace_duration = {}
    spanChildrenMap = {}

    if traceId not in mm_root_map.keys():
        return None

    # add root node
    root_pspan_id = "-1"
    root_ossid = mm_root_map[traceId]['ossid']
    root_nodeid = mm_root_map[traceId]['nodeid']
    root_cmdid = mm_root_map[traceId]['cmdid']
    root_span_id = root_nodeid + root_ossid + root_cmdid
    root_code = mm_root_map[traceId]['code']
    root_start_time = int(time.mktime(time.strptime(
        mm_root_map[traceId]['start_time'], "%Y-%m-%d %H:%M:%S")))
    root_service_name = get_service_name(root_ossid)
    root_duration = int(mm_root_map[traceId]['cost_time'])
    if root_service_name == "":
        root_service_name = root_ossid

    # add root info
    root = Span({
        ITEM.TRACE_ID: traceId,
        ITEM.SPAN_ID: root_span_id,
        ITEM.PARENT_SPAN_ID: root_pspan_id,
        ITEM.START_TIME: root_start_time,
        ITEM.DURATION: root_duration,
        ITEM.SERVICE: root_ossid,
        ITEM.OPERATION: root_cmdid,
        ITEM.SPAN_TYPE: 'EntrySpan',
        ITEM.PEER: "{}/{}".format(root_service_name, root_cmdid),
        ITEM.CODE: root_code,
        ITEM.IS_ERROR: False,
    })
    trace.insert(0, root)
    spanChildrenMap[root.spanId] = []

    # generate span dict
    for span in trace:
        spanMap[span.spanId] = span
        if span.parentSpanId not in spanChildrenMap.keys():
            spanChildrenMap[span.parentSpanId] = []
        spanChildrenMap[span.parentSpanId].append(span)

    # process other span
    for span in trace:
        """
        (raph object contains Vertexs and Edges
        Edge: [(from, to, duration), ...]
        Vertex: [(id, nodestr), ...]
        """

        # get the parent server span id
        if span.parentSpanId == '-1':
            rootSpan = span
            trace_duration["start"] = span.startTime
            trace_duration["end"] = span.startTime + \
                span.duration + 1 if span.duration <= 0 else 0
        else:
            if spanMap.get(span.parentSpanId) is None:
                if span.parentSpanId in spanChildrenMap.keys():
                    del spanChildrenMap[span.parentSpanId]
                span.parentSpanId = root.spanId
                spanChildrenMap[root.spanId].append(span)

        if span.parentSpanId not in spanIdMap.keys():
            spanIdMap[span.parentSpanId] = spanIdCounter
            spanIdCounter += 1

        if span.spanId not in spanIdMap.keys():
            spanIdMap[span.spanId] = spanIdCounter
            spanIdCounter += 1

        vid, pvid = spanIdMap[span.spanId], spanIdMap[span.parentSpanId]

        # span id should be unique
        if vid not in vertexs.keys():
            opname = '/'.join([span.service, span.operation])
            vertexs[vid] = [span.service, opname]

        if pvid not in edges.keys():
            edges[str(pvid)] = []

        feats = calculate_edge_features(
            span, trace_duration, spanChildrenMap)
        feats['vertexId'] = vid
        feats['duration'] = time_normolize(span.duration)

        if span.operation not in operation_map.keys():
            operation_map[span.operation] = {}
            for key in operation_select_keys:
                operation_map[span.operation][key] = []
        for key in operation_select_keys:
            operation_map[span.operation][key].append(feats[key])

        edges[str(pvid)].append(feats)

    if rootSpan == None:
        return None

    if len(edges) > 1000 or len(vertexs) < 2:
        return None

    graph = {
        'abnormal': 0,
        'rc': '',
        'vertexs': vertexs,
        'edges': edges,
    }

    return graph


def get_mmapi() -> dict:
    api_file = './secrets/api.yaml'
    print(f"read api url from {api_file}")

    with open(api_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data


def load_name_cache() -> dict:
    with open(cache_file, 'r') as f:
        cache = json.load(f)
        print(f"load cache from {cache_file}")

    return cache


def save_data(graphs: Dict, idx: str = ''):
    """
    save graph data to json file
    """
    dir = 'trainticket'
    if dtype == DataType.Wechat:
        dir = 'wechat'
    elif dtype == DataType.AIops:
        dir = 'aiops'

    filepath = utils.generate_save_filepath('data.json', time_now_str, dir)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    print("saving data..., map size: {}".format(sys.getsizeof(graphs)))
    with open(filepath, 'w', encoding='utf-8') as fd:
        json.dump(graphs, fd, ensure_ascii=False)

    print(f"{len(graphs)} traces data saved in {filepath}")


def divide_word(s: str, sep: str = "/") -> str:
    if dtype == DataType.Wechat:
        return sep.join(wordninja.split(s))

    words = ['ticket', 'order', 'name', 'security',
             'operation', 'spring', 'service', 'trip',
             'date', 'route', 'type', 'id', 'account', 'number']
    word_list = []
    s = s.replace('-', '/')
    s = s.replace('_', '/')
    s = s.replace('{', '')
    s = s.replace('}', '')
    s = s.lower()
    s = s.strip('/')

    for w in s.split('/'):
        for sub in utils.wordSplit(w, words):
            snake = utils.hump2snake(sub)
            word_list.append(snake)

    return sep.join(word_list)


def trace_process(trace: List[Span], enable_word_division: bool) -> List[Span]:
    peerMap = {}
    for span in trace:
        if span.spanType == "Exit":
            peerMap[span.parentSpanId] = span.peer

    for span in trace:
        if span.spanType == "Entry" and span.spanId in peerMap.keys():
            span.peer = peerMap[span.spanId]
    return trace


def get_operation_name(cmdid: int, module_name: str) -> str:
    global cache

    if module_name == "":
        return str(cmdid)

    if module_name not in cache['cmd_name'].keys():
        cache['cmd_name'][module_name] = {}

    if cmdid in cache['cmd_name'][module_name].keys():
        return cache['cmd_name'][module_name][cmdid]

    if use_request:
        params = {
            'sn': sn,
            'fields': 'interface_id,name,module_id,module_name,interface_id',
            'page': 1,
            'page_size': 1000,
            'where_module_name': module_name,
            'where_interface_id': cmdid,
        }

        try:
            rsp = requests.get(operation_url, timeout=10, params=params)
        except Exception as e:
            print(f"get operation name from cmdb failed:", e)
        else:
            if rsp.ok:
                datas = rsp.json()['data']
                if len(datas) > 0:
                    name = datas[0]['name']
                    cache['cmd_name'][module_name][cmdid] = name
                    return str(name)
                # not found
                cache['cmd_name'][module_name][cmdid] = str(cmdid)
                return cmdid
            print(f'cant get operation name, code:', rsp.status_code)

    return str(cmdid)


def get_service_name(ossid: int) -> str:
    global cache

    if ossid in cache['oss_name'].keys():
        return str(cache['oss_name'][ossid])

    if use_request:
        params = {
            'sn': sn,
            'fields': 'module_name,ossid,module_id',
            'where_ossid': ossid,
        }

        try:
            rsp = requests.get(service_url, timeout=10, params=params)
        except Exception as e:
            print(f"get service name from cmdb failed:", e)
        else:
            if rsp.ok:
                datas = rsp.json()['data']
                if len(datas) > 0:
                    name = str(datas[0]['module_name'])
                    cache['oss_name'][ossid] = name
                    return name
                # not found
                cache['oss_name'][ossid] = str(ossid)
                return ""
            print(f'cant get name, code:', rsp.status_code)

    return ""


def save_name_cache(cache: dict):
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=4)
        print('save cache success')


def glove_embedding() -> Callable[[str], List[float]]:
    embedding_word_list = np.load('./data/glove/wordsList.npy').tolist()
    embedding_word_vector = np.load('./data/glove/wordVectors.npy')

    def glove(input: str) -> List[float]:
        words = input.split('/')
        vec_sum = []
        for w in words:
            if w in embedding_word_list:
                idx = embedding_word_list.index(w)
                vec = embedding_word_vector[idx]
                vec_sum.append(vec)

        return np.mean(np.array(vec_sum), axis=0).tolist()

    return glove


def bert_embedding() -> Callable[[str], List[float]]:
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=True, cache_dir='./data/cache'
    )

    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, cache_dir='./data/cache'
    )

    def bert(input: str) -> List[float]:
        inputs = tokenizer(
            input, padding='max_length', max_length=100, return_tensors="pt")

        outputs = model(**inputs)

        return outputs.pooler_output.tolist()[0]

    return bert


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


def task(ns, idx, divide_word: bool = True):
    span_data = ns.sl[idx]
    current = current_process()
    pos = current._identity[0] - 1
    graph_map = {}
    operation_map = {}
    for trace_id, trace_data in tqdm(span_data.groupby([ITEM.TRACE_ID]), desc="processing #{:0>2d}".format(idx),
                                     position=pos):
        trace = [Span(raw_span) for _, raw_span in trace_data.iterrows()]
        graph = build_graph(trace_process(
            trace, divide_word), normalize, operation_map)
        if graph == None:
            continue
        graph_map[trace_id] = graph

    return graph_map


# use for data cache
dataset = []


def preprocess_span(start: int, end: int, stage: str) -> dict:
    """
    获取毫秒时间戳start~end之间的span, 保存为data.json
    返回一个data dict
    """
    global dataset
    if len(dataset) == 0:
        dataset = load_span(stage)
    win_spans = []
    for df in dataset:
        ss = df.loc[(df.StartTime > start) & (df.StartTime < end)]
        if len(ss) > 0:
            win_spans.append(ss)

    if len(win_spans) <= 0:
        return {}

    result_map = {}

    # With shared memory
    with Manager() as m:
        ns = m.Namespace()
        ns.sl = win_spans
        with ProcessPoolExecutor(cpu_count()) as exe:
            data_size = len(win_spans)
            fs = [exe.submit(task, ns, idx, False)
                  for idx in range(data_size)]
            for fu in as_completed(fs):
                graphs = fu.result()
                result_map = utils.mergeDict(result_map, graphs)

    if len(result_map) > 0:
        return result_map

    return {}


def main():
    args = arguments()
    global dtype, use_request, embedding_name, rm_non_rc_abnormal
    if args.wechat:
        dtype = DataType.Wechat
    if args.aiops:
        dtype = DataType.AIops
    rm_non_rc_abnormal = args.rm_non_rc_abnormal
    use_request = args.use_request

    print(f"parallel processing number: {args.cores}")

    # load all span
    raw_spans = load_span(dtype)
    # if is_wechat and use_request:
    #     save_name_cache(cache)

    # concat all span data in one list
    # span_data = pd.concat(raw_spans, axis=0, ignore_index=True)

    # global normalize
    # if args.normalize == 'minmax':
    #     max_duration = span_data[ITEM.DURATION].max()
    #     min_duration = span_data[ITEM.DURATION].min()

    #     def normalize(x):
    #         return min_max(
    #             x, min_duration, max_duration)

    # elif args.normalize == 'zscore':
    #     mean_duration = span_data[ITEM.DURATION].mean()
    #     std_duration = span_data[ITEM.DURATION].std()

    #     def normalize(x):
    #         return z_score(
    #             x, mean_duration, std_duration)

    # del span_data

    # global embedding
    # if embedding_name == 'glove':
    #     embedding = glove_embedding()
    #     enable_word_division = True
    # elif embedding_name == 'bert':
    #     embedding = bert_embedding()
    #     enable_word_division = False
    # else:
    #     print(f"invalid embedding method name: {embedding_name}")
    #     exit()

    result_map = {}

    # With shared memory
    with Manager() as m:
        ns = m.Namespace()
        ns.sl = raw_spans
        with ProcessPoolExecutor(args.cores) as exe:
            data_size = len(raw_spans)
            fs = [exe.submit(task, ns, idx)
                  for idx in range(data_size)]
            for fu in as_completed(fs):
                graphs = fu.result()
                result_map = utils.mergeDict(result_map, graphs)

                # control the data size
                # if len(result_map) > args.max_num:
                #     save_data(result_map, str(file_idx))
                #     file_idx = file_idx + 1
                #     result_map = {}

    if len(result_map) > 0:
        save_data(result_map)

    # print('start generate embedding file')
    # name_dict = {}
    # for name in tqdm(name_set):
    #     name_dict[name] = embedding(name)

    # embd_filepath = utils.generate_save_filepath(
    #     'embeddings.json', time_now_str, is_wechat)
    # with open(embd_filepath, 'w', encoding='utf-8') as fd:
    #     json.dump(name_dict, fd, ensure_ascii=False)
    # print(f'embedding data saved in {embd_filepath}')

    # operation_filepath = utils.generate_save_filepath(
    #     'operations.json', time_now_str, is_wechat)
    # with open(operation_filepath, 'w', encoding='utf-8') as fo:
    #     json.dump(operation_map, fo, ensure_ascii=False)
    # print(f'operations data saved in {operation_filepath}')

    # print('preprocess finished :)')


if __name__ == '__main__':
    main()
    # def timestamp(datetime) -> int:
    #     timeArray = time.strptime(datetime, "%Y-%m-%d %H:%M:%S")
    #     ts = int(time.mktime(timeArray)) * 1000
    #     # print(ts)
    #     return ts

    # start = '2022-02-27 01:00:00'
    # end = '2022-02-27 01:30:00'

    # res = preprocess_span(start=timestamp(start), end=timestamp(end))
    # print(res)

    print('preprocess finished :)')
