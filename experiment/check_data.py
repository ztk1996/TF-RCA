import json


file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-05_16-38-16/data.json', 'r')
data = json.load(file)
dset = [['ts-rebook-service'], 1651664892776, 1651665198006]

count = 0
for tid, trace in data.items():
	if trace['abnormal'] == 1 and trace['edges']['0'][0]['startTime'] > dset[1] and trace['edges']['0'][0]['startTime'] < dset[2]:
		count += 1
		root = trace['rc'][0]
		print(tid, root)
		nodes = []
		# for v in trace['vertexs'].values():
		# 	nodes.append(v[0])
		for es in trace['edges'].values():
			for e in es:
				nodes.append(e['service'])
		print('nodes:', nodes)


print(count)