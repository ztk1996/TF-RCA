import json
import numpy as np
import pandas as pd


seq_set = []
with open('.data/seq_set.json', 'r') as fd:
	seq_set = json.load(fd)

stat = {}
with open('.data/stat.json', 'r') as fd:
	stat = json.load(fd)


test_result = pd.read_csv('./result_real_data/test.csv')

raw_data = {}
with open('/data/TraceCluster/raw/trainticket/chaos/..') as fd:
	raw_data = json.laod(fd)

