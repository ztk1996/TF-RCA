# preprocess global parameters

new_data_list = [
    # New Data

    # Normal
    # 'normal/2022-01-11_00-00-00_12h_traces.csv',
    # 'normal/2022-01-11_12-00-00_6h_traces.csv',

    # Chaos
    # 'chaos/2022-01-10_16-00-00_5h_traces.csv',
    # 'chaos/2022-01-11_00-00-00_24h_traces.csv',
    # 'chaos/2022-01-12_00-00-00_24h_traces.csv',
    'chaos/2022-01-13_00-00-00_24h_traces.csv',
    # 'chaos/2022-02-27_00-00-00_12h_traces.csv',
    # 'chaos/2022-03-01_11-00-00_3h_traces.csv'
]

old_data_list = [
    # Normal
    'normal/normal0822_01/SUCCESS_SpanData2021-08-22_15-43-01.csv',
    'normal/normal0822_02/SUCCESS2_SpanData2021-08-22_22-04-35.csv',
    'normal/normal0823/SUCCESS2_SpanData2021-08-23_15-15-08.csv',

    # F01
    'F01-01/SUCCESSF0101_SpanData2021-08-14_10-22-48.csv',
    'F01-02/ERROR_F012_SpanData2021-08-14_01-52-43.csv',
    'F01-03/SUCCESSerrorf0103_SpanData2021-08-16_16-17-08.csv',
    'F01-04/SUCCESSF0104_SpanData2021-08-14_02-14-51.csv',
    'F01-05/SUCCESSF0105_SpanData2021-08-14_02-45-59.csv',

    # F02
    'F02-01/SUCCESS_errorf0201_SpanData2021-08-17_18-25-59.csv',
    'F02-02/SUCCESS_errorf0202_SpanData2021-08-17_18-47-04.csv',
    'F02-03/SUCCESS_errorf0203_SpanData2021-08-17_18-54-53.csv',
    'F02-04/ERROR_SpanData.csv',
    'F02-05/ERROR_SpanData.csv',
    'F02-06/ERROR_SpanData.csv',

    # F03
    'F03-01/ERROR_SpanData.csv',
    'F03-02/ERROR_SpanData.csv',
    'F03-03/ERROR_SpanData.csv',
    'F03-04/ERROR_SpanData.csv',
    'F03-05/ERROR_SpanData.csv',
    'F03-06/ERROR_SpanData.csv',
    'F03-07/ERROR_SpanData.csv',
    'F03-08/ERROR_SpanData.csv',

    # F04
    'F04-01/ERROR_SpanData.csv',
    'F04-02/ERROR_SpanData.csv',
    'F04-03/ERROR_SpanData.csv',
    'F04-04/ERROR_SpanData.csv',
    'F04-05/ERROR_SpanData.csv',
    'F04-06/ERROR_SpanData.csv',
    'F04-07/ERROR_SpanData.csv',
    'F04-08/ERROR_SpanData.csv',

    # F05
    "F05-02/ERROR_errorf0502_SpanData2021-08-10_13-53-38.csv",
    "F05-03/ERROR_SpanData2021-08-07_20-34-09.csv",
    "F05-04/ERROR_SpanData2021-08-07_21-02-22.csv",
    "F05-05/ERROR_SpanData2021-08-07_21-28-23.csv",

    # F07
    "F07-01/back0729/ERROR_SpanData2021-07-29_10-36-21.csv",
    "F07-01/back0729/SUCCESS_SpanData2021-07-29_10-38-09.csv",
    "F07-01/ERROR_errorf0701_SpanData2021-08-10_14-09-59.csv",
    "F07-02/back0729/ERROR_SpanData2021-07-29_13-58-37.csv",
    "F07-02/back0729/SUCCESS_SpanData2021-07-29_13-51-48.csv",
    "F07-02/ERROR_errorf0702_SpanData2021-08-10_14-33-35.csv",
    "F07-03/ERROR_SpanData2021-08-07_22-53-33.csv",
    "F07-04/ERROR_SpanData2021-08-07_23-49-11.csv",
    "F07-05/ERROR_SpanData2021-08-07_23-57-44.csv",

    # F08
    "F08-01/ERROR_SpanData2021-07-29_19-15-36.csv",
    "F08-01/SUCCESS_SpanData2021-07-29_19-16-01.csv",
    "F08-02/ERROR_SpanData2021-07-30_10-13-04.csv",
    "F08-02/SUCCESS_SpanData2021-07-30_10-13-46.csv",
    "F08-03/ERROR_SpanData2021-07-30_12-07-36.csv",
    "F08-03/SUCCESS_SpanData2021-07-30_12-07-23.csv",
    "F08-04/ERROR_SpanData2021-07-30_14-20-15.csv",
    "F08-04/SUCCESS_SpanData2021-07-30_14-22-24.csv",
    "F08-05/ERROR_SpanData2021-07-30_11-00-30.csv",
    "F08-05/SUCCESS_SpanData2021-07-30_11-01-05.csv",

    # F11
    "F11-01/SUCCESSF1101_SpanData2021-08-14_10-18-35.csv",
    "F11-02/SUCCESSerrorf1102_SpanData2021-08-16_16-57-36.csv",
    "F11-03/SUCCESSF1103_SpanData2021-08-14_03-04-11.csv",
    "F11-04/SUCCESSF1104_SpanData2021-08-14_03-35-38.csv",
    "F11-05/SUCCESSF1105_SpanData2021-08-14_03-38-35.csv",

    # F12
    "F12-01/ERROR_SpanData2021-08-12_16-17-46.csv",
    "F12-02/ERROR_SpanData2021-08-12_16-24-54.csv",
    "F12-03/ERROR_SpanData2021-08-12_16-36-33.csv",
    "F12-04/ERROR_SpanData2021-08-12_17-04-34.csv",
    "F12-05/ERROR_SpanData2021-08-12_16-49-08.csv",

    # F13
    "F13-01/SUCCESSerrorf1301_SpanData2021-08-16_21-01-36.csv",
    "F13-02/SUCCESS_SpanData2021-08-13_17-34-58.csv",
    "F13-03/SUCCESSerrorf1303_SpanData2021-08-16_18-55-52.csv",
    "F13-04/SUCCESSF1304_SpanData2021-08-14_10-50-42.csv",
    "F13-05/SUCCESSF1305_SpanData2021-08-14_11-13-43.csv",

    # F14
    "F14-01/SUCCESS_SpanData2021-08-12_14-56-41.csv",
    "F14-02/SUCCESS_SpanData2021-08-12_15-24-50.csv",
    "F14-03/SUCCESS_SpanData2021-08-12_15-46-08.csv",

    # F23
    "F23-01/ERROR_SpanData2021-08-07_20-30-26.csv",
    "F23-02/ERROR_SpanData2021-08-07_20-51-14.csv",
    "F23-03/ERROR_SpanData2021-08-07_21-10-11.csv",
    "F23-04/ERROR_SpanData2021-08-07_21-34-47.csv",
    "F23-05/ERROR_SpanData2021-08-07_22-02-42.csv",

    # F24
    "F24-01/ERROR_SpanData.csv",
    "F24-02/ERROR_SpanData.csv",
    "F24-03/ERROR_SpanData.csv",

    # F25
    "F25-01/ERROR_SpanData2021-08-16_11-17-21.csv",
    "F25-02/ERROR_SpanData2021-08-16_11-21-59.csv",
    "F25-03/ERROR_SpanData2021-08-16_12-20-59.csv",
]

# skywalking data list
data_path_list = [
    # *old_data_list,
    *new_data_list,
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

# wechat trace root data list
mm_trace_root_list = [
    # '11-29/click_stream_2021-11-29_23629.csv',
    # '12-3/click_stream_2021-12-03_24486.csv',
    # '12-3/click_stream_2021-12-03_23629.csv',
    # 'trace_mmfindersynclogicsvr/click_stream_2022-01-17_23629.csv',
    'trace_mmfindersynclogicsvr/click_stream_2022-01-18_23629.csv',
]

# see https://openmsg.yuque.com/openmsg/wechat/smhizg#1Gpq
chaos_dict = {
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

service_changes = [
    # normal changes
    ('ts-route-service', 'codewisdom/ts-route-service:0.2.0'),
    ('ts-order-service', 'codewisdom/ts-order-service:0.2.0'),
    ('ts-auth-service', 'codewisdom/ts-auth-service:0.2.0'),

    # abnormal changes
    ('ts-ticketinfo-service', 'codewisdom/ts-ticketinfo-service:0.2.0'),
    ('ts-travel-service', 'codewisdom/ts-travel-service:0.2.0'),
    ('ts-route-service', 'codewisdom/ts-route-service:0.2.0'),
    ('ts-order-service', 'codewisdom/ts-order-service:0.2.0'),
    ('ts-auth-service', 'codewisdom/ts-auth-service:0.2.0'),
    ('ts-user-service', 'codewisdom/ts-user-service:0.2.0'),
]

change_order1 = [
    [1], [0], [0], [2], [1],
    [2], [1], [1], [2], [4],
    [0], [0], [4], [0], [2],
    [1], [5], [0], [0], [1],
    [5], [7], [5], [4], [3],
    [5], [2], [2], [4], [4],
    [1], [1], [0], [4], [2],
    [6], [6], [7], [0], [3],
    [0], [5], [5], [1], [5],
    [8], [1], [0], [2], [5],
]

change_order2 = [
    [1, 0], [4, 5], [0, 1], [1, 0], [2, 8],
    [4, 5], [1, 7], [2, 0], [0, 1], [7, 8],
    [0, 3], [3, 1], [0, 0], [8, 2], [2, 1],
    [1, 7], [8, 7], [1, 0], [8, 0], [0, 5],
    [7, 4], [4, 2], [3, 4], [8, 0], [0, 1],
    [1, 2], [3, 7], [1, 2], [0, 3], [0, 0],
    [6, 8], [1, 2], [3, 2], [5, 0], [2, 0],
    [6, 3], [2, 2], [4, 8], [2, 1], [8, 7],
    [1, 0], [4, 7], [0, 5], [5, 0], [0, 2],
    [0, 2], [0, 0], [1, 1], [4, 6], [8, 2],
]

request_period_log = [
    ([1], 1649929146713, 1649929194306), ([0], 1649929314769, 1649929341460), ([0], 1649929361570, 1649929397046), ([2], 1649929497398, 1649929508477), ([1], 1649929628956, 1649929648660), ([2], 1649929759061, 1649929774470), ([1], 1649929894832, 1649929907123), ([1], 1649929927241, 1649929942142), ([2], 1649930052497, 1649930063152), ([4], 1649930143418, 1649930154147), ([0], 1649930264508, 1649930291119), ([0], 1649930311250, 1649930327727), ([4], 1649930408025,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1649930419498), ([0], 1649930439604, 1649930454114), ([2], 1649930564464, 1649930574860), ([1], 1649930695238, 1649930714433), ([5], 1649930834826, 1649930845857), ([0], 1649930865979, 1649930882789), ([0], 1649930902912, 1649930918389), ([1], 1649931028739, 1649931042368), ([5], 1649931162722, 1649931175568), ([7], 1649931285901, 1649931300329), ([5], 1649931420734, 1649931433282), ([4], 1649931513542, 1649931528384), ([3], 1649931638714, 1649931651307),
    ([5], 1649931771652, 1649931785455), ([2], 1649931895779, 1649931910354), ([2], 1649931930465, 1649931941771), ([4], 1649932022020, 1649932032883), ([4], 1649932052995, 1649932064067), ([1], 1649932184408, 1649932202197), ([1], 1649932222314, 1649932233293), ([0], 1649932353646, 1649932365387), ([4], 1649932445636, 1649932455904), ([2], 1649932556224, 1649932571182), ([6], 1649932691657, 1649932706325), ([6], 1649932726440, 1649932746192), ([7], 1649932856570,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1649932872105), ([0], 1649932982520, 1649932997307), ([3], 1649933107683, 1649933128778), ([0], 1649933249137, 1649933267005), ([5], 1649933287123, 1649933300785), ([5], 1649933320900, 1649933337201), ([1], 1649933447506, 1649933457559), ([5], 1649933577995, 1649933589356), ([8], 1649933699707, 1649933712014), ([1], 1649933832402, 1649933847532), ([0], 1649933967872, 1649933985405), ([2], 1649934085750, 1649934098222), ([5], 1649934208538, 1649934220453)
]
