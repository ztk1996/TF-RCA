import pandas as pd
import numpy as np

# chaos = [['ts-station-service'], 1650980195250, 1650980499254]
filepath = '/data/TraceCluster/raw/trainticket/chaos/2022-04-28_23-00-00_6h_traces.csv'

data_type = {'StartTime': np.uint64, 'EndTime': np.uint64}
spans = pd.read_csv(filepath, dtype=data_type).drop_duplicates().dropna()

spans['Duration'] = spans['EndTime'] - spans['StartTime']
# df = spans.loc[(spans['StartTime'] > chaos[1]) & (spans['StartTime'] < chaos[2])]
df = spans.loc[spans['TraceId'] == '6bb6d6cdad40414e880b0537e5977559.38.16511592677620001']
df.to_csv('tmp.csv')