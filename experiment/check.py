import pandas as pd
import numpy as np

chaos = [['ts-station-service', 'ts-user-service'], 1650823144636, 1650823444945]
filepath = '/data/TraceCluster/raw/trainticket/chaos/2022-04-24_19-00-00_8h_traces.csv'

data_type = {'StartTime': np.uint64, 'EndTime': np.uint64}
spans = pd.read_csv(filepath, dtype=data_type).drop_duplicates().dropna()

spans['Duration'] = spans['EndTime'] - spans['StartTime']
# df = spans.loc[(spans['StartTime'] > chaos[1]) & (spans['StartTime'] < chaos[2]) & (spans['Duration'] > 5000)]
df = spans.loc[spans['TraceId'] == '475bf33c8c3a4fef8e9da155775f9e95.36.16508231673300081']
df.to_csv('tmp.csv')