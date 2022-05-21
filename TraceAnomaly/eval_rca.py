import json
import numpy as np
import pandas as pd
import random

ta_root = './TraceAnomaly/'
# request_period_log = [(['ts-seat-service'], 1651748486185, 1651748797147), (['ts-basic-service'], 1651748987478, 1651749295867), (['ts-ticketinfo-service'], 1651749486209, 1651749795299), (['ts-order-other-service'], 1651749985634, 1651750293217), (['ts-consign-service'], 1651750483565, 1651750791138), (['ts-price-service'], 1651750981502, 1651751287378), (['ts-travel-service'], 1651751477713, 1651751902504), (['ts-route-service'], 1651752092854, 1651752394833), (['ts-travel-service'], 1651752585171, 1651752893040), (['ts-train-service'], 1651753083378, 1651753394162), (['ts-config-service'], 1651753584488, 1651753888468), (['ts-order-service'], 1651754078802, 1651754385570), (['ts-route-service'], 1651754575912, 1651754881391), (['ts-train-service'], 1651755071738, 1651755376210), (['ts-user-service'], 1651755566548, 1651755874017), (['ts-route-service'], 1651756064360, 1651756365927), (['ts-ticketinfo-service'], 1651756556284, 1651756859350), (['ts-price-service'], 1651757049709, 1651757354076), (['ts-basic-service'], 1651757544405, 1651757851873), (['ts-user-service'], 1651758042207, 1651758348752), (['ts-order-other-service'], 1651758539096, 1651758846454), (['ts-price-service'], 1651759036801, 1651759343752), (['ts-order-other-service'], 1651759534094, 1651759839642), (['ts-travel-plan-service'], 1651760029972, 1651760337711), (['ts-train-service'], 1651760528062, 1651760834111), (['ts-config-service'], 1651761024450, 1651761331805), (['ts-rebook-service'], 1651761522155, 1651761825288), (['ts-consign-service'], 1651762015613, 1651762323166), (['ts-order-service'], 1651762513510, 1651762820468), (['ts-route-service'], 1651763010837, 1651763316689), (['ts-rebook-service'], 1651763507033, 1651763866086), (['ts-station-service'], 1651764056441, 1651764363577), (['ts-ticketinfo-service'], 1651764553920, 1651764859280), (['ts-travel-service'], 1651765049611, 1651765355247), (['ts-seat-service'], 1651765545589, 1651765851820), (['ts-rebook-service'], 1651766042163, 1651766349694), (['ts-station-service'], 1651766540051, 1651766845284), (['ts-config-service'], 1651767035612, 1651767337932), (['ts-consign-service'], 1651767528273, 1651767835018), (['ts-station-service'], 1651768025355, 1651768329778), (['ts-order-service'], 1651768520117, 1651768825535), (['ts-basic-service'], 1651769015884, 1651769317814), (['ts-travel-plan-service'], 1651769508162, 1651769810980), (['ts-seat-service'], 1651770001315, 1651770307832), (['ts-user-service'], 1651770498181, 1651770805499), (['ts-basic-service'], 1651770995821, 1651771303663), (['ts-basic-service'], 1651771494025, 1651771796434), (['ts-basic-service'], 1651771986783, 1651772289847), (['ts-order-service'], 1651772480173, 1651772786498), (['ts-order-service'], 1651772976840, 1651773281769)]
request_period_log=[(['ts-user-service'], 1652082509749, 1652083413427), (['ts-order-service'], 1652083663660, 1652084565731), (['ts-ticketinfo-service'], 1652085969552, 1652086874918), (['ts-user-service'], 1652089451055, 1652090354006), (['ts-order-service'], 1652091758536, 1652092661425), (['ts-route-service'], 1652092841565, 1652093769877), (['ts-travel-service'], 1652094250394, 1652095155270), (['ts-order-service'], 1652095335407, 1652096237776), (['ts-route-service'], 1652096488003, 1652097391167), (['ts-travel-service'], 1652100191897, 1652101095460), (['ts-route-service'], 1652102519903, 1652103423257), (['ts-user-service'], 1652103603394, 1652104537479), (['ts-order-service'], 1652106039521, 1652106995920), (['ts-order-service'], 1652107176058, 1652108144193), (['ts-ticketinfo-service'], 1652108394417, 1652109344989), (['ts-user-service'],
					1652109525119, 1652110531415), (['ts-travel-service'], 1652112178207, 1652113127882), (['ts-route-service'], 1652113378101, 1652114336493), (['ts-auth-service'], 1652114586728, 1652115591876), (['ts-route-service'], 1652115772008, 1652116725466), (['ts-user-service'], 1652116975688, 1652117962611), (['ts-auth-service'], 1652119424828, 1652120364654), (['ts-order-service'], 1652122941765, 1652123950378), (['ts-order-service'], 1652127697299, 1652128601830), (['ts-ticketinfo-service'], 1652132342339, 1652133244902), (['ts-route-service'], 1652133495128, 1652134473369), (['ts-route-service'], 1652134723596, 1652135654796), (['ts-auth-service'], 1652135905018, 1652136830032), (['ts-order-service'], 1652137080255, 1652137983164), (['ts-route-service'], 1652138233396, 1652139203956), (['ts-route-service'], 1652139384092, 1652140289239)]

def read_raw_vector(input_file):  # flows, vectors, valid_column
    with open(input_file, 'r') as fin:
        raw = fin.read().strip().split('\n')

    vectors = {}
    for line in raw:
        if line.strip() == "":
            continue
        vectors[line.split(':')[0]] = [float(x) for x in line.split(':')[1].split(',')]

    return vectors

def main():
	print('load seq set...')
	seq_set = []
	with open(ta_root + 'data/seq_set.json', 'r') as fd:
		seq_set = json.load(fd)

	print('load stat...')
	stat = {}
	with open(ta_root + 'data/stat.json', 'r') as fd:
		stat = json.load(fd)

	print('load test result...')
	test_result = pd.read_csv(ta_root + 'result_real_data/test.csv')

	print('load raw data...')
	raw_data = {}
	with open('/home/kagaya/work/TF-RCA/data/preprocessed/trainticket/2022-05-11_00-06-32/data.json') as fd:
		raw_data = json.load(fd)

	# 获取每条异常trace的根因排名
	a_df = test_result.loc[test_result['pred'] == 1]
	a_trace_ids = a_df['id'].tolist()

	print('load vector...')
	vectors = read_raw_vector(ta_root + 'data/test_normal')
	vectors.update(read_raw_vector(ta_root + 'data/test_abnormal'))

	a_trace_path = {}
	for tid in a_trace_ids:
		a_path = {}
		v = vectors[tid]
		for i in range(len(seq_set)):
			duration = v[i]
			if duration == 0:
				continue
			path = seq_set[i]
			
			if path not in stat.keys():
				if path not in a_path.keys():
					a_path[path] = 0
				a_path[path] += 1
			else:
				mean, std = stat[path][0], stat[path][1]
				if duration < mean-3*std or duration > mean+3*std:
					if path not in a_path.keys():
						a_path[path] = 0
					a_path[path] += 1
		
		a_trace_path[tid] = a_path

	cases = [{} for _ in range(len(request_period_log))]
	for tid in a_trace_ids:
		graph = raw_data[tid]
		path_count = a_trace_path[tid]
		start_time = int(graph['edges']['0'][0]['startTime'])
		i = 0

		for i in range(len(request_period_log)):
			if request_period_log[i][1] < start_time and start_time < request_period_log[i][2]:
				break
		
		for path, count in path_count.items():
			if path not in cases[i].keys():
				cases[i][path] = count
			else:
				cases[i][path] += count

	a_trace_root_rank = []

	for case in cases:
		root_count = {}
		for k, v in case.items():
			root = k.split('->')[-1]
			if root not in root_count:
				root_count[root] = 0
			root_count[root] += v
		a_trace_root_rank.append(sorted(root_count.items(), key = lambda kv: kv[1], reverse=True))

	top_count = [0 for _ in range(4)]
	for i in range(len(a_trace_root_rank)):
		case = a_trace_root_rank[i][:10]
		root = request_period_log[i][0][0]
		for j in range(len(case)):
			if root == case[j][0]:
				if j < 1:
					top_count[0] += 1
				if j < 3:
					top_count[1] += 1
				if j < 5:
					top_count[2] += 1
				if j < 7:
					top_count[3] += 1

	# print(f'case {i} top-10:', case)
	print('top count:', top_count)

if __name__ == '__main__':
	main()