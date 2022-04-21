# preprocess global parameters

new_data_list = [
    # New Data

    # Normal
    # 'normal/2022-01-11_00-00-00_12h_traces.csv',
    # 'normal/2022-01-11_12-00-00_6h_traces.csv',
    'normal/2022-04-20_00-00-00_10h_traces.csv'

    # Chaos
    # 'chaos/2022-04-16_20-00-00_8h_traces.csv',
    # 'chaos/2022-04-18_11-00-00_8h_traces.csv',
    # 'chaos/2022-04-18_21-00-00_6h_traces.csv',
    # 'chaos/2022-04-19_09-00-00_9h_traces.csv',
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

# 2022-4-16 abnormal
# request_period_log = [(['ts-route-service'], 1650110947285, 1650111022655), (['ts-travel-service'], 1650111443255, 1650111516641), (['ts-travel-service'], 1650111937224, 1650112009193), (['ts-travel-service'], 1650112429798, 1650112660081), (['ts-station-service'], 1650112920542, 1650112983912), (['ts-basic-service'], 1650113414514, 1650113717655), (['ts-ticketinfo-service'], 1650113908061, 1650113989462), (['ts-user-service'], 1650114400046, 1650114461963), (['ts-order-service'], 1650114892565, 1650114987733), (['ts-order-service'], 1650115388402, 1650115442406), (['ts-order-service'], 1650115883030, 1650115923986), (['ts-basic-service'], 1650116374616, 1650116429343), (['ts-route-service'], 1650116869960, 1650117080792), (['ts-user-service'], 1650117361254, 1650117408133), (['ts-station-service'], 1650117858740, 1650117903699), (['ts-ticketinfo-service'], 1650118354344, 1650118404123), (['ts-travel-plan-service'], 1650118854750, 1650118952633), (['ts-travel-plan-service'], 1650119353203, 1650119420244), (['ts-user-service'], 1650119850850, 1650119902530), (['ts-basic-service'], 1650120343133, 1650120413145), (['ts-ticketinfo-service'], 1650120833760, 1650121136454), (['ts-station-service'], 1650121326847, 1650121390708), (['ts-travel-plan-service'], 1650121821330, 1650121886301), (['ts-route-service'], 1650122316895, 1650122380688), (['ts-travel-service'], 1650122811324, 1650123158723), (['ts-route-service'], 1650123426438, 1650123487715), (['ts-basic-service'], 1650123918328, 1650124174477), (['ts-travel-service'], 1650124414927, 1650124512745), (['ts-user-service'], 1650124913317, 1650124979527), (['ts-ticketinfo-service'], 1650125410142, 1650125468870), (['ts-travel-service'], 1650125909496, 1650125962648), (['ts-order-service'], 1650126403277, 1650126476160), (['ts-station-service'], 1650126896736, 1650126928726), (['ts-user-service'], 1650127389357, 1650127445357), (['ts-order-service'], 1650127885960, 1650127937873), (['ts-travel-plan-service'], 1650128378504, 1650128458527), (['ts-order-service'], 1650128869114, 1650128924960), (['ts-basic-service'], 1650129365571, 1650129418909), (['ts-basic-service'], 1650129859518, 1650129973511), (['ts-route-service'], 1650130354074, 1650130510388), (['ts-station-service'], 1650130850916, 1650130943772), (['ts-ticketinfo-service'], 1650131344334, 1650131375970), (['ts-station-service'], 1650131836580, 1650131917762), (['ts-route-service'], 1650132328369, 1650132414588), (['ts-travel-plan-service'], 1650132825168, 1650132870912), (['ts-user-service'], 1650133321535, 1650133372570), (['ts-ticketinfo-service'], 1650133813202, 1650133873973), (['ts-travel-plan-service'], 1650134304574, 1650134335995), (['ts-order-service'], 1650134796598, 1650134877487), (['ts-travel-service'], 1650135288084, 1650135344221)]

# 2022-4-17 abnormal2
# request_period_log = [(['ts-station-service'], 1650171990302, 1650172105396), (['ts-travel-plan-service'], 1650172105398, 1650172192504), (['ts-travel-plan-service'], 1650172483296, 1650172694285), (['ts-ticketinfo-service'], 1650172694286, 1650172824950), (['ts-travel-plan-service'], 1650173015640, 1650173124532), (['ts-user-service'], 1650173124533, 1650173155918), (['ts-route-service'], 1650173506744, 1650173697989), (['ts-user-service'], 1650173697990, 1650173728924), (['ts-order-service'], 1650173999718, 1650174041484), (['ts-route-service'], 1650174041485, 1650174109287), (['ts-travel-service'], 1650174500144, 1650174664461), (['ts-ticketinfo-service'], 1650174664462, 1650174740447), (['ts-travel-plan-service'], 1650174991165, 1650175041197), (['ts-station-service'], 1650175041198, 1650175084419), (['ts-station-service'], 1650175485315, 1650175555334), (['ts-route-service'], 1650175555335, 1650175611493), (['ts-user-service'], 1650175982320, 1650176025188), (['ts-order-service'], 1650176025189, 1650176077659), (['ts-order-service'], 1650176478572, 1650176547669), (['ts-basic-service'], 1650176547670, 1650176581457), (['ts-ticketinfo-service'], 1650176972322, 1650177030242), (['ts-travel-plan-service'], 1650177030243, 1650177101133), (['ts-ticketinfo-service'], 1650177472005, 1650177696340), (['ts-travel-service'], 1650177696341, 1650177798472), (['ts-travel-service'], 1650177989146, 1650178200183), (['ts-basic-service'], 1650178200184, 1650178306255), (['ts-route-service'], 1650178496941, 1650178568385), (['ts-order-service'], 1650178568386, 1650178622301), (['ts-route-service'], 1650178993161, 1650179066254), (['ts-ticketinfo-service'], 1650179066255, 1650179105701), (['ts-basic-service'], 1650179486583, 1650179697421), (['ts-route-service'], 1650179697422, 1650179810182), (['ts-ticketinfo-service'], 1650180000872, 1650180081939), (['ts-travel-service'], 1650180081940, 1650180115543), (['ts-basic-service'], 1650180496383, 1650180552243), (['ts-basic-service'], 1650180552244, 1650180615437), (['ts-travel-service'], 1650180996247, 1650181094157), (['ts-order-service'], 1650181094158, 1650181175275), (['ts-user-service'], 1650181496065, 1650181588884), (['ts-station-service'], 1650181588884, 1650181674820), (['ts-basic-service'], 1650181995613, 1650182098461), (['ts-travel-service'], 1650182098462, 1650182174483), (['ts-station-service'], 1650182495286, 1650182711741), (['ts-user-service'], 1650182711742, 1650182774584), (['ts-user-service'], 1650182995297, 1650183056254), (['ts-station-service'], 1650183056255, 1650183266815), (['ts-order-service'], 1650183487529, 1650183574879), (['ts-travel-plan-service'], 1650183574880, 1650183630726), (['ts-ticketinfo-service'], 1650183981606, 1650184023864), (['ts-travel-service'], 1650184023865, 1650184103268),
#                       (['ts-user-service'], 1650184474128, 1650184540498), (['ts-ticketinfo-service'], 1650184540498, 1650184610174), (['ts-station-service'], 1650184971038, 1650185075081), (['ts-ticketinfo-service'], 1650185075081, 1650185131504), (['ts-ticketinfo-service'], 1650185462308, 1650185577259), (['ts-travel-service'], 1650185577260, 1650185656095), (['ts-route-service'], 1650185956918, 1650186019582), (['ts-order-service'], 1650186019583, 1650186076982), (['ts-route-service'], 1650186447842, 1650186511801), (['ts-user-service'], 1650186511802, 1650186577973), (['ts-order-service'], 1650186938810, 1650187047753), (['ts-station-service'], 1650187047754, 1650187086290), (['ts-ticketinfo-service'], 1650187437202, 1650187507358), (['ts-basic-service'], 1650187507359, 1650187552110), (['ts-order-service'], 1650187932955, 1650187975669), (['ts-route-service'], 1650187975670, 1650188022494), (['ts-basic-service'], 1650188433356, 1650188644092), (['ts-ticketinfo-service'], 1650188644093, 1650188747657), (['ts-travel-service'], 1650188938403, 1650189058902), (['ts-order-service'], 1650189058903, 1650189115024), (['ts-user-service'], 1650189435831, 1650189466817), (['ts-travel-service'], 1650189466817, 1650189673546), (['ts-travel-plan-service'], 1650189934282, 1650190043128), (['ts-travel-plan-service'], 1650190043129, 1650190098855), (['ts-travel-service'], 1650190429659, 1650190588884), (['ts-basic-service'], 1650190588885, 1650190740255), (['ts-user-service'], 1650190930941, 1650190998187), (['ts-user-service'], 1650190998188, 1650191038976), (['ts-travel-plan-service'], 1650191429846, 1650191483405), (['ts-travel-plan-service'], 1650191483406, 1650191536708), (['ts-travel-service'], 1650191927561, 1650192061699), (['ts-user-service'], 1650192061699, 1650192093406), (['ts-station-service'], 1650192424245, 1650192483851), (['ts-station-service'], 1650192483852, 1650192522029), (['ts-basic-service'], 1650192922880, 1650192986673), (['ts-station-service'], 1650192986674, 1650193039469), (['ts-station-service'], 1650193420338, 1650193581531), (['ts-route-service'], 1650193581532, 1650193732684), (['ts-order-service'], 1650193923408, 1650194002337), (['ts-route-service'], 1650194002338, 1650194121133), (['ts-travel-plan-service'], 1650194421906, 1650194540736), (['ts-basic-service'], 1650194540737, 1650194584391), (['ts-basic-service'], 1650194915184, 1650195126006), (['ts-order-service'], 1650195126007, 1650195206978), (['ts-route-service'], 1650195407660, 1650195574122), (['ts-travel-plan-service'], 1650195574122, 1650195629825), (['ts-station-service'], 1650195900594, 1650196001282), (['ts-user-service'], 1650196001283, 1650196046884), (['ts-travel-plan-service'], 1650196397735, 1650196495355), (['ts-travel-plan-service'], 1650196495356, 1650196528820)]

# 2022-4-18 11:00:00 8h abnormal
# request_period_log = [(['ts-route-service'], 1650254059896, 1650254361387), (['ts-travel-service'], 1650254551823, 1650254853456), (['ts-travel-service'], 1650255043883, 1650255344428), (['ts-travel-service'], 1650255534831, 1650255851210), (['ts-station-service'], 1650256151142, 1650256452270), (['ts-basic-service'], 1650256642674, 1650256944234), (['ts-ticketinfo-service'], 1650257134636, 1650257435240), (['ts-user-service'], 1650257625637, 1650257927939), (['ts-order-service'], 1650258118339, 1650258421608), (['ts-order-service'], 1650258611989, 1650258912546), (['ts-order-service'], 1650259102938, 1650259403142), (['ts-basic-service'], 1650259593530, 1650259896864), (['ts-route-service'], 1650260087260, 1650260388746), (['ts-user-service'], 1650260579154, 1650260881122), (['ts-station-service'], 1650261071509, 1650261371783), (['ts-ticketinfo-service'], 1650261562192, 1650261865276), (['ts-travel-plan-service'], 1650262055678, 1650262358656), (['ts-travel-plan-service'], 1650262549050, 1650262852081), (['ts-user-service'], 1650263042497, 1650263343295), (['ts-basic-service'], 1650263533699, 1650263836890), (['ts-ticketinfo-service'], 1650264027284, 1650264328269), (['ts-station-service'], 1650264518660, 1650264821500), (['ts-travel-plan-service'], 1650265011904, 1650265312147), (['ts-route-service'], 1650265502560, 1650265803476), (['ts-travel-service'], 1650265993869, 1650266305823),
#                       (['ts-route-service'], 1650266496226, 1650266796479), (['ts-basic-service'], 1650266986888, 1650267287653), (['ts-travel-service'], 1650267478060, 1650267779310), (['ts-user-service'], 1650267969704, 1650268271204), (['ts-ticketinfo-service'], 1650268461603, 1650268763257), (['ts-travel-service'], 1650268953663, 1650269257054), (['ts-order-service'], 1650269447456, 1650269749062), (['ts-station-service'], 1650269939458, 1650270241764), (['ts-user-service'], 1650270432172, 1650270732294), (['ts-order-service'], 1650270922698, 1650271225238), (['ts-travel-plan-service'], 1650271415635, 1650271715726), (['ts-order-service'], 1650271906150, 1650272206462), (['ts-basic-service'], 1650272396851, 1650272699948), (['ts-basic-service'], 1650272890362, 1650273192473), (['ts-route-service'], 1650273382897, 1650273684256), (['ts-station-service'], 1650273874658, 1650274177836), (['ts-ticketinfo-service'], 1650274368244, 1650274668476), (['ts-station-service'], 1650274858860, 1650275159546), (['ts-route-service'], 1650275349948, 1650275652839), (['ts-travel-plan-service'], 1650275843231, 1650276144522), (['ts-user-service'], 1650276334921, 1650276636243), (['ts-ticketinfo-service'], 1650276826642, 1650277128245), (['ts-travel-plan-service'], 1650277318638, 1650277619725), (['ts-order-service'], 1650277810111, 1650278110838), (['ts-travel-service'], 1650278301231, 1650278626536)]

# 2022-4-18 21:00:00 6h change
# request_period_log = [(['ts-ticketinfo-service'], 1650287471783, 1650287776660), (['ts-order-service'], 1650287876778, 1650288181809), (['ts-route-service'], 1650288281931, 1650288663553), (['ts-auth-service'], 1650288763672, 1650289064272), (['ts-auth-service'], 1650289565214, 1650289865310), (['ts-ticketinfo-service'], 1650290834575, 1650291139408), (['ts-order-service'], 1650291239528, 1650291540143), (['ts-route-service'], 1650292046194, 1650292378922), (['ts-user-service'], 1650292479042, 1650292783835), (['ts-order-service'], 1650292883958, 1650293201857), (['ts-route-service'], 1650293301977, 1650293667144), (['ts-order-service'], 1650294169400, 1650294476553), (['ts-route-service'], 1650294576673, 1650294886480), (['ts-order-service'], 1650295790136, 1650296095290), (['ts-route-service'], 1650296195415, 1650296495666), (['ts-user-service'], 1650296595786, 1650296899568), (['ts-order-service'], 1650296999686, 1650297303330), (['ts-route-service'], 1650298288435, 1650298594501), (['ts-ticketinfo-service'], 1650299911473, 1650300214161), (['ts-order-service'], 1650300719828, 1650301023083), (['ts-route-service'], 1650301123200, 1650301444469), (['ts-user-service'], 1650301544589, 1650301848409), (['ts-travel-service'], 1650301948529, 1650302253335), (['ts-ticketinfo-service'], 1650302353482, 1650302662805), (['ts-order-service'], 1650302762923, 1650303071956), (['ts-ticketinfo-service'], 1650303172077, 1650303481396), (['ts-order-service'], 1650303581516, 1650303887312), (['ts-order-service'], 1650303987433, 1650304292240), (['ts-order-service'], 1650305260968, 1650305569179), (['ts-route-service'], 1650306480421, 1650306859404), (['ts-order-service'], 1650306959524, 1650307260489)]

# 2022-4-19 9:00:00 abnormal2
request_period_log = [(['ts-station-service'], 1650336179212, 1650336338648), (['ts-travel-plan-service'], 1650336338650, 1650336489436), (['ts-travel-plan-service'], 1650336680107, 1650336872914), (['ts-ticketinfo-service'], 1650336872915, 1650337025399), (['ts-travel-plan-service'], 1650337216099, 1650337368320), (['ts-user-service'], 1650337368321, 1650337520949), (['ts-route-service'], 1650337711647, 1650337871546), (['ts-user-service'], 1650337871547, 1650338023361), (['ts-order-service'], 1650338214052, 1650338366435), (['ts-route-service'], 1650338366436, 1650338518370), (['ts-travel-service'], 1650338709071, 1650338861748), (['ts-ticketinfo-service'], 1650338861749, 1650339013901), (['ts-travel-plan-service'], 1650339204678, 1650339361041), (['ts-station-service'], 1650339361042, 1650339514958), (['ts-station-service'], 1650339705647, 1650339857074), (['ts-route-service'], 1650339857075, 1650340011826), (['ts-user-service'], 1650340202506, 1650340355037), (['ts-order-service'], 1650340355038, 1650340509414), (['ts-order-service'], 1650340700095, 1650340851954), (['ts-basic-service'], 1650340851955, 1650341004961), (['ts-ticketinfo-service'], 1650341195642, 1650341349690), (['ts-travel-plan-service'], 1650341349691, 1650341501773), (['ts-ticketinfo-service'], 1650341692441, 1650341861850), (['ts-travel-service'], 1650341861851, 1650342012116), (['ts-travel-service'], 1650342202815, 1650342367111), (['ts-basic-service'], 1650342367112, 1650342518881), (['ts-route-service'], 1650342709594, 1650342864176), (['ts-order-service'], 1650342864177, 1650343015730), (['ts-route-service'], 1650343206439, 1650343363239), (['ts-ticketinfo-service'], 1650343363239, 1650343518137), (['ts-basic-service'], 1650343708829, 1650343901509), (['ts-route-service'], 1650343901510, 1650344052381), (['ts-ticketinfo-service'], 1650344243091, 1650344397776), (['ts-travel-service'], 1650344397776, 1650344550633), (['ts-basic-service'], 1650344741304, 1650344895215), (['ts-basic-service'], 1650344895216, 1650345050365), (['ts-travel-service'], 1650345241033, 1650345394373), (['ts-order-service'], 1650345394374, 1650345548447), (['ts-user-service'], 1650345739152, 1650345890971), (['ts-station-service'], 1650345890972, 1650346043886), (['ts-basic-service'], 1650346234547, 1650346386840), (['ts-travel-service'], 1650346386840, 1650346539141), (['ts-station-service'], 1650346729817, 1650346922374), (['ts-user-service'], 1650346922375, 1650347075810), (['ts-user-service'], 1650347266548, 1650347419580), (['ts-station-service'], 1650347419581, 1650347570932), (['ts-order-service'], 1650347761650, 1650347919692), (['ts-travel-plan-service'], 1650347919693, 1650348071006), (['ts-ticketinfo-service'], 1650348261694, 1650348421751), (['ts-travel-service'], 1650348421752, 1650348574513), (['ts-user-service'], 1650348765223, 1650348917391), (['ts-ticketinfo-service'], 1650348917392, 1650349070689), (['ts-station-service'], 1650349261451, 1650349415723), (['ts-ticketinfo-service'], 1650349415724, 1650349568402), (['ts-ticketinfo-service'], 1650349759093, 1650349927981), (['ts-travel-service'], 1650349927982, 1650350080539), (['ts-route-service'], 1650350271229, 1650350424738), (['ts-order-service'], 1650350424740, 1650350577413), (['ts-route-service'], 1650350768090, 1650350918949), (['ts-user-service'], 1650350918950, 1650351070299), (['ts-order-service'], 1650351260983, 1650351416964), (['ts-station-service'], 1650351416965, 1650351567646), (['ts-ticketinfo-service'], 1650351758326, 1650351912652), (['ts-basic-service'], 1650351912653, 1650352065473), (['ts-order-service'], 1650352256184, 1650352407200), (['ts-route-service'], 1650352407201, 1650352560273), (['ts-basic-service'], 1650352750999, 1650352943714), (['ts-ticketinfo-service'], 1650352943715, 1650353095615), (['ts-travel-service'], 1650353286373, 1650353468136), (['ts-order-service'], 1650353468137, 1650353620494), (['ts-user-service'], 1650353811210, 1650353963495), (['ts-travel-service'], 1650353963496, 1650354113743), (['ts-travel-plan-service'], 1650354304423, 1650354458425), (['ts-travel-plan-service'], 1650354458426, 1650354611981), (['ts-travel-service'], 1650354802657, 1650354971060), (['ts-basic-service'], 1650354971061, 1650355121343), (['ts-user-service'], 1650355312044, 1650355465001), (['ts-user-service'], 1650355465002, 1650355616292), (['ts-travel-plan-service'], 1650355806995, 1650355961650), (['ts-travel-plan-service'], 1650355961651, 1650356113964), (['ts-travel-service'], 1650356304634, 1650356473369), (['ts-user-service'], 1650356473370, 1650356625693), (['ts-station-service'], 1650356816397, 1650356973753), (['ts-station-service'], 1650356973754, 1650357127193), (['ts-basic-service'], 1650357317866, 1650357480777), (['ts-station-service'], 1650357480778, 1650357634702), (['ts-station-service'], 1650357825428, 1650358017911), (['ts-route-service'], 1650358017912, 1650358169783), (['ts-order-service'], 1650358360468, 1650358510817), (['ts-route-service'], 1650358510818, 1650358661637), (['ts-travel-plan-service'], 1650358852344, 1650359009771), (['ts-basic-service'], 1650359009772, 1650359161949), (['ts-basic-service'], 1650359352625, 1650359512018), (['ts-order-service'], 1650359512019, 1650359666037), (['ts-route-service'], 1650359856740, 1650360011058), (['ts-travel-plan-service'], 1650360011059, 1650360161811), (['ts-station-service'], 1650360352506, 1650360544451), (['ts-user-service'], 1650360544452, 1650360696892), (['ts-travel-plan-service'], 1650360887596, 1650361039265), (['ts-travel-plan-service'], 1650361039265, 1650361193550)]