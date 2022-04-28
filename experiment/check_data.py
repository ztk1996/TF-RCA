import json


file = open(r'/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-04-27_22-19-20/data.json', 'r')
data = json.load(file)

count = 0
for tid, trace in data.items():
	if trace['abnormal'] == 1 and trace['edges']['0'][0]['startTime'] > 1650978000000 and trace['edges']['0'][0]['startTime'] < 1650978360000:
		count += 1
		root = trace['rc'][0]
		print(tid, root)
		nodes = []
		for v in trace['vertexs'].values():
			nodes.append(v[0])
		print(nodes)

print(count)