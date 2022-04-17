# preprocess global parameters

new_data_list = [
    # New Data

    # Normal
    # 'normal/2022-01-11_00-00-00_12h_traces.csv',
    # 'normal/2022-01-11_12-00-00_6h_traces.csv',

    # Chaos
    'chaos/2022-04-16_20-00-00_8h_traces.csv',
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


request_period_log = [(['ts-route-service'], 1650110947285, 1650111022655), (['ts-travel-service'], 1650111443255, 1650111516641), (['ts-travel-service'], 1650111937224, 1650112009193), (['ts-travel-service'], 1650112429798, 1650112660081), (['ts-station-service'], 1650112920542, 1650112983912), (['ts-basic-service'], 1650113414514, 1650113717655), (['ts-ticketinfo-service'], 1650113908061, 1650113989462), (['ts-user-service'], 1650114400046, 1650114461963), (['ts-order-service'], 1650114892565, 1650114987733), (['ts-order-service'], 1650115388402, 1650115442406), (['ts-order-service'], 1650115883030, 1650115923986), (['ts-basic-service'], 1650116374616, 1650116429343), (['ts-route-service'], 1650116869960, 1650117080792), (['ts-user-service'], 1650117361254, 1650117408133), (['ts-station-service'], 1650117858740, 1650117903699), (['ts-ticketinfo-service'], 1650118354344, 1650118404123), (['ts-travel-plan-service'], 1650118854750, 1650118952633), (['ts-travel-plan-service'], 1650119353203, 1650119420244), (['ts-user-service'], 1650119850850, 1650119902530), (['ts-basic-service'], 1650120343133, 1650120413145), (['ts-ticketinfo-service'], 1650120833760, 1650121136454), (['ts-station-service'], 1650121326847, 1650121390708), (['ts-travel-plan-service'], 1650121821330, 1650121886301), (['ts-route-service'], 1650122316895, 1650122380688), (['ts-travel-service'], 1650122811324, 1650123158723),
                      (['ts-route-service'], 1650123426438, 1650123487715), (['ts-basic-service'], 1650123918328, 1650124174477), (['ts-travel-service'], 1650124414927, 1650124512745), (['ts-user-service'], 1650124913317, 1650124979527), (['ts-ticketinfo-service'], 1650125410142, 1650125468870), (['ts-travel-service'], 1650125909496, 1650125962648), (['ts-order-service'], 1650126403277, 1650126476160), (['ts-station-service'], 1650126896736, 1650126928726), (['ts-user-service'], 1650127389357, 1650127445357), (['ts-order-service'], 1650127885960, 1650127937873), (['ts-travel-plan-service'], 1650128378504, 1650128458527), (['ts-order-service'], 1650128869114, 1650128924960), (['ts-basic-service'], 1650129365571, 1650129418909), (['ts-basic-service'], 1650129859518, 1650129973511), (['ts-route-service'], 1650130354074, 1650130510388), (['ts-station-service'], 1650130850916, 1650130943772), (['ts-ticketinfo-service'], 1650131344334, 1650131375970), (['ts-station-service'], 1650131836580, 1650131917762), (['ts-route-service'], 1650132328369, 1650132414588), (['ts-travel-plan-service'], 1650132825168, 1650132870912), (['ts-user-service'], 1650133321535, 1650133372570), (['ts-ticketinfo-service'], 1650133813202, 1650133873973), (['ts-travel-plan-service'], 1650134304574, 1650134335995), (['ts-order-service'], 1650134796598, 1650134877487), (['ts-travel-service'], 1650135288084, 1650135344221)]
