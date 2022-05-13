import pandas as pd
import numpy as np

chaos = [['ts-order-service'], 1651772480173, 1651772786498]
filepath = '/data/TraceCluster/raw/trainticket/chaos/2022-05-05_19-00-00_8h_traces.csv'

data_type = {'StartTime': np.uint64, 'EndTime': np.uint64}
spans = pd.read_csv(filepath, dtype=data_type).drop_duplicates().dropna()

spans['Duration'] = spans['EndTime'] - spans['StartTime']
# df = spans.loc[(spans['StartTime'] > chaos[1]) & (spans['StartTime'] < chaos[2]) & ((spans['Peer'] == chaos[0][0]) | (spans['Service'] == chaos[0][0]))]
df = spans.loc[(spans['Code'] == 404) & ((spans['Peer'] == chaos[0][0]) | (spans['Service'] == chaos[0][0]))]
# df = spans.loc[spans['TraceId'] == '6bb6d6cdad40414e880b0537e5977559.38.16511592677620001']
df.to_csv('tmp.csv')