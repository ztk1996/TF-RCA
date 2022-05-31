# preprocess global parameters


from requests import request


data_root = '/data/TraceCluster/raw'

new_data_list = [
    # Normal
    # 'normal/2022-04-20_00-00-00_10h_traces.csv'
    # 'normal/2022-05-03_20-00-00_14h_traces.csv',
    # 'normal/2022-05-07_08-00-00_8h_traces.csv',

    # Chaos

    # add new cases
    # 'chaos/2022-05-04_16-00-00_8h_traces.csv',  # 1 abnormal
    # 'chaos/2022-05-04_16-00-00_8h_traces.csv', # 2 abnormal
    # 'chaos/2022-05-05_19-00-00_8h_traces.csv',  # 1 abnormal

    # 'chaos/2022-05-07_15-00-00_8h_traces.csv',  # 1 abnormal
    # 'chaos/2022-05-09_15-00-00_17h_traces.csv'  # new case 1 change

    # 'chaos/2022-04-27_15-00-00_8h_traces.csv'    # 1 abnormal avail
    # 'chaos/2022-05-01_00-00-00_9h_traces.csv',    # 1 change
    # 'chaos/2022-04-28_12-00-00_8h_traces.csv',    # 2 abnormal
    # 'chaos/2022-05-05_19-00-00_8h_traces.csv',    # 1 abnormal new 5-6

]


aiops_data_list = [
    '2020_05_31/trace/trace_csf.csv',
    '2020_05_31/trace/trace_jdbc.csv',
    '2020_05_31/trace/trace_osb.csv',
    '2020_05_31/trace/trace_fly_remote.csv',
    '2020_05_31/trace/trace_local.csv',
    '2020_05_31/trace/trace_remote_process.csv',
]

init_data_path = [
    # 'normal/2022-04-20_00-00-00_10h_traces.csv',
    # 'normal/2022-05-03_20-00-00_14h_traces.csv'
    ]

# wechat data list
mm_data_path_list = [
    # '5-18/finer_data.json',
    # '5-18/finer_data2.json',
    # '8-2/data.json',
    # '8-3/data.json',
    # '11-9/data.json',
    # '11-18/call_graph_2021-11-18_61266.csv'
    # '11-22/call_graph_2021-11-22_23629.csv'
    # '11-29/call_graph_2021-11-29_23629.csv',
    # '12-3/call_graph_2021-12-03_24486.csv',
    # '12-3/call_graph_2021-12-03_23629.csv',
    # 'trace_mmfindersynclogicsvr/call_graph_2022-01-17_23629.csv',
    'trace_mmfindersynclogicsvr/call_graph_2022-01-18_23629.csv',
    # 'trace_mmfindersynclogicsvr/call_graph_18_500000.csv',
]

init_mm_data_path_list = [
    # '5-18/finer_data.json',
    # '5-18/finer_data2.json',
    # '8-2/data.json',
    # '8-3/data.json',
    # '11-9/data.json',
    # '11-18/call_graph_2021-11-18_61266.csv'
    # '11-22/call_graph_2021-11-22_23629.csv'
    # '11-29/call_graph_2021-11-29_23629.csv',
    # '12-3/call_graph_2021-12-03_24486.csv',
    # '12-3/call_graph_2021-12-03_23629.csv',
    # 'trace_mmfindersynclogicsvr/call_graph_2022-01-17_23629.csv',
    'trace_mmfindersynclogicsvr/call_graph_2022-01-18_23629.csv',
    # 'trace_mmfindersynclogicsvr/call_graph_18_500000.csv',
]

# wechat trace root data list
mm_trace_root_list = [
    # '11-29/click_stream_2021-11-29_23629.csv',
    # '12-3/click_stream_2021-12-03_24486.csv',
    # '12-3/click_stream_2021-12-03_23629.csv',
    # 'trace_mmfindersynclogicsvr/click_stream_2022-01-17_23629.csv',
    'trace_mmfindersynclogicsvr/click_stream_2022-01-18_23629.csv',
]

# data list
data_path_list = [
    # *new_data_list,
    *mm_data_path_list,
]

init_data_path_list = [
    # *new_data_list,
    *mm_data_path_list,
]

# see https://openmsg.yuque.com/openmsg/wechat/smhizg#1Gpq
span_chaos_dict = {
    0: 'ts-travel-service',
    1: 'ts-ticketinfo-service',
    2: 'ts-route-service',
    3: 'ts-order-service',
    4: 'ts-basic-service',
    5: 'ts-basic-service',
    6: 'ts-travel-plan-service',
    7: 'ts-station-service',
    8: 'ts-seat-service',
    9: 'ts-config-service',
    10: 'ts-inside-payment-service',
    11: 'ts-cancel-service',
    12: 'ts-contacts-service',
    13: 'ts-consign-service',
    14: 'ts-consign-price-service',
    15: 'ts-auth-service',
    16: 'ts-execute-service',
    17: 'ts-preserve-service',
    18: 'ts-user-service',
    19: 'ts-user-service',
}

request_period_log = []

# abnormal 1 new 5-6
# request_period_log = [(['ts-seat-service'], 1651748486185, 1651748797147), (['ts-basic-service'], 1651748987478, 1651749295867), (['ts-ticketinfo-service'], 1651749486209, 1651749795299), (['ts-order-other-service'], 1651749985634, 1651750293217), (['ts-consign-service'], 1651750483565, 1651750791138), (['ts-price-service'], 1651750981502, 1651751287378), (['ts-travel-service'], 1651751477713, 1651751902504), (['ts-route-service'], 1651752092854, 1651752394833), (['ts-travel-service'], 1651752585171, 1651752893040), (['ts-train-service'], 1651753083378, 1651753394162), (['ts-config-service'], 1651753584488, 1651753888468), (['ts-order-service'], 1651754078802, 1651754385570), (['ts-route-service'], 1651754575912, 1651754881391), (['ts-train-service'], 1651755071738, 1651755376210), (['ts-user-service'], 1651755566548, 1651755874017), (['ts-route-service'], 1651756064360, 1651756365927), (['ts-ticketinfo-service'], 1651756556284, 1651756859350), (['ts-price-service'], 1651757049709, 1651757354076), (['ts-basic-service'], 1651757544405, 1651757851873), (['ts-user-service'], 1651758042207, 1651758348752), (['ts-order-other-service'], 1651758539096, 1651758846454), (['ts-price-service'], 1651759036801, 1651759343752), (['ts-order-other-service'], 1651759534094, 1651759839642), (['ts-travel-plan-service'], 1651760029972, 1651760337711), (['ts-train-service'], 1651760528062, 1651760834111), (['ts-config-service'], 1651761024450, 1651761331805), (['ts-rebook-service'], 1651761522155, 1651761825288), (['ts-consign-service'], 1651762015613, 1651762323166), (['ts-order-service'], 1651762513510, 1651762820468), (['ts-route-service'], 1651763010837, 1651763316689), (['ts-rebook-service'], 1651763507033, 1651763866086), (['ts-station-service'], 1651764056441, 1651764363577), (['ts-ticketinfo-service'], 1651764553920, 1651764859280), (['ts-travel-service'], 1651765049611, 1651765355247), (['ts-seat-service'], 1651765545589, 1651765851820), (['ts-rebook-service'], 1651766042163, 1651766349694), (['ts-station-service'], 1651766540051, 1651766845284), (['ts-config-service'], 1651767035612, 1651767337932), (['ts-consign-service'], 1651767528273, 1651767835018), (['ts-station-service'], 1651768025355, 1651768329778), (['ts-order-service'], 1651768520117, 1651768825535), (['ts-basic-service'], 1651769015884, 1651769317814), (['ts-travel-plan-service'], 1651769508162, 1651769810980), (['ts-seat-service'], 1651770001315, 1651770307832), (['ts-user-service'], 1651770498181, 1651770805499), (['ts-basic-service'], 1651770995821, 1651771303663), (['ts-basic-service'], 1651771494025, 1651771796434), (['ts-basic-service'], 1651771986783, 1651772289847), (['ts-order-service'], 1651772480173, 1651772786498), (['ts-order-service'], 1651772976840, 1651773281769)]

# change 1 new 5-10
# request_period_log=[(['ts-user-service'], 1652082509749, 1652083413427), (['ts-order-service'], 1652083663660, 1652084565731), (['ts-ticketinfo-service'], 1652085969552, 1652086874918), (['ts-user-service'], 1652089451055, 1652090354006), (['ts-order-service'], 1652091758536, 1652092661425), (['ts-route-service'], 1652092841565, 1652093769877), (['ts-travel-service'], 1652094250394, 1652095155270), (['ts-order-service'], 1652095335407, 1652096237776), (['ts-route-service'], 1652096488003, 1652097391167), (['ts-travel-service'], 1652100191897, 1652101095460), (['ts-route-service'], 1652102519903, 1652103423257), (['ts-user-service'], 1652103603394, 1652104537479), (['ts-order-service'], 1652106039521, 1652106995920), (['ts-order-service'], 1652107176058, 1652108144193), (['ts-ticketinfo-service'], 1652108394417, 1652109344989), (['ts-user-service'],
#                      1652109525119, 1652110531415), (['ts-travel-service'], 1652112178207, 1652113127882), (['ts-route-service'], 1652113378101, 1652114336493), (['ts-auth-service'], 1652114586728, 1652115591876), (['ts-route-service'], 1652115772008, 1652116725466), (['ts-user-service'], 1652116975688, 1652117962611), (['ts-auth-service'], 1652119424828, 1652120364654), (['ts-order-service'], 1652122941765, 1652123950378), (['ts-order-service'], 1652127697299, 1652128601830), (['ts-ticketinfo-service'], 1652132342339, 1652133244902), (['ts-route-service'], 1652133495128, 1652134473369), (['ts-route-service'], 1652134723596, 1652135654796), (['ts-auth-service'], 1652135905018, 1652136830032), (['ts-order-service'], 1652137080255, 1652137983164), (['ts-route-service'], 1652138233396, 1652139203956), (['ts-route-service'], 1652139384092, 1652140289239)]
normal_change_log = [(['ts-food-map-service'], 1652081354241, 1652082259526), (['ts-food-map-service'], 1652084815947, 1652085719325), (['ts-order-service'], 1652087135157, 1652088038731), (['ts-travel-service'], 1652088298966, 1652089200832), (['ts-order-service'], 1652090604236, 1652091508310), (['ts-food-map-service'], 1652097641388, 1652098546854), (['ts-route-plan-service'], 1652098807105, 1652099711365), (['ts-travel-service'], 1652101365722, 1652102269670), (['ts-route-service'], 1652104807735, 1652105759259), (['ts-route-plan-service'],
                    1652110791661, 1652111697663), (['ts-route-service'], 1652118222855, 1652119174603), (['ts-food-map-service'], 1652120614871, 1652121553285), (['ts-route-service'], 1652121803510, 1652122761635), (['ts-food-map-service'], 1652124200602, 1652125139823), (['ts-travel-service'], 1652125390050, 1652126292791), (['ts-order-service'], 1652126543021, 1652127447071), (['ts-route-plan-service'], 1652128882085, 1652129785216), (['ts-food-map-service'], 1652130035429, 1652130938454), (['ts-route-plan-service'], 1652131188674, 1652132092114)]


trace_chaos_dict = {
    0: 'ts-payment-service',    # pod fail
    1: 'ts-ticketinfo-service',  # http delay
    2: 'ts-route-service',  # network delay
    3: 'ts-order-service',  # network delay
    4: 'ts-basic-service',  # pod fail
    5: 'ts-basic-service',  # http delay
    6: 'ts-ticketinfo-service',  # pod failure
    8: 'ts-seat-service',   # pod failure
    9: 'ts-config-service',  # http delay
    10: 'ts-travel-service',    # pod failure
    11: 'ts-order-other-service',   # http delay
    12: 'ts-order-other-service',   # pod failure
    13: 'ts-consign-price-service',  # http delay
    14: 'ts-consign-price-service',  # http delay
    15: 'ts-verification-code-service',  # pod failure
    16: 'ts-order-service',  # pod failure
    17: 'ts-security-service',  # http delay
    18: 'ts-verification-code-service',  # http delay
    19: 'ts-user-service',  # http delay
    20: 'ts-station-service',   # pod fail
    21: 'ts-route-service',  # pod failure
    22: 'ts-travel2service',    # pod kill
    23: 'ts-config-service'  # pod failure
}