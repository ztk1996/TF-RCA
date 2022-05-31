import json
from tqdm import tqdm

dirpath = '/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-11_00-06-32/'

with open(dirpath + 'data.json', 'r') as fd:
    data = json.load(fd)

normal = {}
abnormal = {}
for k, v in tqdm(data.items()):
    if v['abnormal'] == 1:
        abnormal[k] = v
    else:
        normal[k] = v

with open(dirpath + 'data_normal.json', 'w') as fd:
    json.dump(normal, fd)


with open(dirpath + 'data_abnormal.json', 'w') as fd:
    json.dump(abnormal, fd)