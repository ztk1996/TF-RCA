import json


file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-29_23-08-54/data.json', 'r')
data = json.load(file)
dset = [['ts-user-service'], 1651163469669, 1651163770339]

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
		print(nodes)


print(count)