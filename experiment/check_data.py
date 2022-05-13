import json


file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-06_17-28-43/data.json', 'r')
data = json.load(file)
dset = [['ts-seat-service'], 1651770001315, 1651770307832]

count = 0
for tid, trace in data.items():
	if trace['abnormal'] == 1 and trace['edges']['0'][0]['startTime'] > dset[1] and trace['edges']['0'][0]['startTime'] < dset[2]:
		count += 1
		root = trace['rc']
		print(tid, root)
		nodes = []
		# for v in trace['vertexs'].values():
		# 	nodes.append(v[0])
		for es in trace['edges'].values():
			for e in es:
				nodes.append(e['service'])
		print('nodes:', nodes)


print(count)