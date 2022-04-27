import pandas as pd
import numpy as np

chaos = [['ts-route-service'], 1650978142584, 1650978445305]
filepath = '/data/TraceCluster/raw/trainticket/chaos/2022-04-26_21-00-00_8h_traces.csv'

data_type = {'StartTime': np.uint64, 'EndTime': np.uint64}
spans = pd.read_csv(filepath, dtype=data_type).drop_duplicates().dropna()

spans['Duration'] = spans['EndTime'] - spans['StartTime']
df = spans.loc[(spans['StartTime'] > chaos[1]) & (spans['StartTime'] < chaos[2])]
# df = spans.loc[spans['TraceId'] == 'dc12f4449e854ad18a56a8feab58bb93.43.16508002299850001']
df.to_csv('tmp.csv')