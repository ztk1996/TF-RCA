import matplotlib.pyplot as plt 
import json
import pickle
from scipy.io import loadmat
import pdb 
import numpy as np
from sklearn.metrics import f1_score
from torch import threshold
'''
rcnn_car = pickle.load(open('/home/tiankun/pro/pytorch-FPN/output/res101/DETRACvoc_test/default/res101_faster_rcnn_iter_120000/car_pr.pkl','rb'))
rcnn_bus = pickle.load(open('/home/tiankun/pro/pytorch-FPN/output/res101/DETRACvoc_test/default/res101_faster_rcnn_iter_120000/bus_pr.pkl','rb'))
rcnn_van = pickle.load(open('/home/tiankun/pro/pytorch-FPN/output/res101/DETRACvoc_test/default/res101_faster_rcnn_iter_120000/van_pr.pkl','rb'))
rcnn_motor = pickle.load(open('/home/tiankun/pro/pytorch-FPN/output/res101/DETRACvoc_test/default/res101_faster_rcnn_iter_120000/motor_pr.pkl','rb'))

with open("yolo_recall.json",'r') as load_f:
    RTK = json.load(load_f)

with open("yolo_precision.json",'r') as load_f:
    PTK = json.load(load_f)
'''
'''
# Faster curve
plt.figure(1)
plt.title('PR Curve')
plt.plot(rcnn_car['rec'],rcnn_car['prec'],color='red',label='car',linewidth=3)
plt.plot(rcnn_bus['rec'],rcnn_bus['prec'],color='limegreen',label='bus',linewidth=3)
plt.plot(rcnn_van['rec'],rcnn_van['prec'],color='skyblue',label='van',linewidth=3)
plt.plot(rcnn_motor['rec'],rcnn_motor['prec'],color='orange',label='motor',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show() 
'''
'''
# YOLO curve
plt.figure(1)
plt.title('PR Curve')
plt.subplot(1,3,1)
plt.plot(RTK['car'],PTK['car'])
plt.subplot(1,3,2)
plt.plot(RTK['bus'],PTK['bus'])
plt.subplot(1,3,3)
plt.plot(RTK['van'],PTK['van'])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
'''
'''
exp1 = loadmat('exp1.mat')
exp3 = loadmat('exp3.mat')
exp5 = loadmat('exp5.mat')
exp5sub1 = loadmat('exp5sub1.mat')
exp5sub2 = loadmat('exp5sub2.mat')
exp6 = loadmat('exp6.mat')

x=np.linspace(0.7,1,30)

plt.figure(1)
plt.title('PRW performance')
plt.plot(x,exp1['map'][::-1],color='red',label='exp1',linewidth=3)
plt.plot(x,exp3['map'][::-1],color='limegreen',label='exp3',linewidth=3)
plt.plot(x,exp5['map'][::-1],color='skyblue',label='exp5',linewidth=3)
plt.plot(x,exp6['map'][::-1],color='orange',label='exp6',linewidth=3)
plt.legend()
plt.xlabel('det_thresh')
plt.ylabel('mAP')
plt.show() 
pdb.set_trace()


plt.figure(2)
plt.title('PR Curve')
plt.plot(RTK['car'],PTK['car'],color='red',label='car',linewidth=3)
plt.plot(RTK['bus'],PTK['bus'],color='limegreen',label='bus',linewidth=3)
plt.plot(RTK['van'],PTK['van'],color='skyblue',label='van',linewidth=3)
plt.plot(RTK['motor'],PTK['motor'],color='orange',label='motor',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()   



#contrast curve
print(rcnn_motor['ap'])
plt.figure(3)
plt.title('Motor PR Curve')
plt.plot(rcnn_motor['rec'],rcnn_motor['prec'],color='red',label='Faster_RCNN',linewidth=3)
plt.plot(RTK['motor'],PTK['motor'],color='limegreen',label='YOLO',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()  

plt.figure(4)
plt.title('Car PR Curve')
plt.plot(rcnn_car['rec'],rcnn_car['prec'],color='red',label='Faster_RCNN',linewidth=3)
plt.plot(RTK['car'],PTK['car'],color='limegreen',label='YOLO',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()  

plt.figure(5)
plt.title('Bus PR Curve')
plt.plot(rcnn_bus['rec'],rcnn_bus['prec'],color='red',label='Faster_RCNN',linewidth=3)
plt.plot(RTK['bus'],PTK['bus'],color='limegreen',label='YOLO',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()  

plt.figure(6)
plt.title('Van PR Curve')
plt.plot(rcnn_van['rec'],rcnn_van['prec'],color='red',label='Faster_RCNN',linewidth=3)
plt.plot(RTK['van'],PTK['van'],color='limegreen',label='YOLO',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()  
'''

# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,2,1)
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(1,2,1)


# x=[0.2*i for i in range(1,16)]
# oim = [i/40. for i in [0, 16, 18, 18, 18, 12, 12, 8, 1, 1, 3, 1, 0, 0, 0]]
# NPSM = [i/40. for i in [0, 24, 26, 26, 26, 20, 20, 12, 2, 2, 6, 2, 0, 0, 0]]
# IAN = [i/40. for i in [0, 31, 33, 33, 33, 30, 30, 17, 4, 4, 8, 4, 2, 2, 2]]
# MGTS = [i/40. for i in [0, 32, 35, 35, 35, 34, 35, 18, 4, 4, 9, 4, 2, 2, 2]]
# CLSA = [i/40. for i in [0, 36, 37, 37, 37, 36, 38, 22, 4, 4, 9, 5, 2, 2, 2]]
# ours_map = [91.73,90.75,86.8,85,81.6,78.5]


# # top-1
# CLSA_top1 =[89.2,88.5,86.7,86.3,85.5,79.4]
# ours_top1 = [93.41,92.9,89.4,88.1,85.1,82.7]

# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# # fig.grid(True)
# fig.plot(x,oim,color='purple',label='Top1',linewidth=1,alpha=1,marker='d')
# fig.plot(x,NPSM,color='limegreen',label='Top3',linewidth=1,alpha=1,marker='8')
# fig.plot(x,IAN,color='orange',label='Top5',linewidth=1,alpha=1,marker='s')
# fig.plot(x,MGTS,color='pink',label='Top7',linewidth=1,alpha=1,marker='*')
# fig.plot(x,CLSA,color='skyblue',label='Top10',linewidth=1,alpha=1,marker='^')
# # fig.plot(x,ours_map,color='red',label='BPNet',linewidth=1,alpha=1,marker='o')
# # legend = fig.legend()
# # legend._remove()
# # fig.legend(loc=1,bbox_to_anchor=(1.05,1.0))
# fig.legend(loc=1,)
# fig.set_xlabel('eps')
# fig.set_ylabel('shooting')
# figall.savefig("single fault.png")
# fig.show()




# RQ1:
# Dataset A: manual label thres / Hit Rate (%)
# threshold=[(1-value)*100 for value in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]]
# HR_1=[rate/41*100 for rate in [18, 18, 18, 19, 19, 20, 20, 22, 24, 23, 26]]
# HR_3=[rate/41*100 for rate in [26, 26, 28, 27, 30, 30, 32, 32, 34, 35, 36]]
# HR_5=[rate/41*100 for rate in [33, 33, 33, 33, 35, 36, 37, 36, 37, 37, 37]]
# HR_7=[rate/41*100 for rate in [35, 35, 35, 35, 36, 37, 37, 38, 37, 38, 38]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(threshold,HR_1,color='blue',label='HR@1',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(threshold,HR_3,color='purple',label='HR@3',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(threshold,HR_5,color='orange',label='HR@5',linewidth=1,alpha=1,marker='*',clip_on=False)
# fig.plot(threshold,HR_7,color='green',label='HR@7',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='lower right', bbox_to_anchor=(1, 0.32), fontsize=13, frameon=False)
# plt.xlim([0, 100])
# plt.xticks([10*i for i in range(0, 11)], fontsize=13)
# plt.ylim([40, 100])
# plt.yticks([10*i for i in range(4, 11)], fontsize=13)
# fig.set_xlabel('Percentage (%)', fontsize=13)
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ1_DatasetA_HitRate.png")

# Dataset A: Anomaly Detection
# threshold=[(1-value)*100 for value in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]]
# f1_score=[value*100 for value in [0.77494, 0.79438, 0.81426, 0.83525, 0.85401, 0.87258, 0.89636, 0.91844, 0.94698, 0.97463, 1]]
# recall=[value*100 for value in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
# precision=[value*100 for value in [0.63257, 0.6589, 0.68671, 0.71711, 0.74521, 0.77396, 0.81219, 0.84918, 0.89931, 0.95051, 1]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(threshold,f1_score,color='blue',label='F1-score',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(threshold,recall,color='orange',label='Recall',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(threshold,precision,color='green',label='Precision',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='lower right', fontsize=13, frameon=False)
# plt.xlim([0, 100])
# plt.xticks([10*i for i in range(0, 11)], fontsize=13)
# plt.yticks(fontsize=13)
# # plt.ylim([30, 90])
# # plt.yticks([10*i for i in range(3, 10)])
# fig.set_xlabel('Percentage (%)', fontsize=13)
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ1_DatasetA_AnomalyDetection.png")

# Dataset B: manual label thres / Hit Rate (%)
# threshold=[(1-value)*100 for value in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]]
# HR_1=[rate/25*100 for rate in [9, 9, 10, 9, 10, 10, 11, 12, 12, 13, 15]]
# HR_3=[rate/25*100 for rate in [17, 17, 17, 18, 17, 20, 19, 21, 22, 22, 24]]
# HR_5=[rate/25*100 for rate in [19, 19, 19, 20, 19, 22, 22, 23, 23, 24, 25]]
# HR_7=[rate/25*100 for rate in [20, 20, 20, 20, 22, 22, 22, 23, 23, 24, 25]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(threshold,HR_1,color='blue',label='HR@1',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(threshold,HR_3,color='purple',label='HR@3',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(threshold,HR_5,color='orange',label='HR@5',linewidth=1,alpha=1,marker='*',clip_on=False)
# fig.plot(threshold,HR_7,color='green',label='HR@7',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='lower right', bbox_to_anchor=(1, 0.4), fontsize=13, frameon=False)
# plt.xlim([0, 100])
# plt.xticks([10*i for i in range(0, 11)], fontsize=13)
# plt.ylim([30, 100])
# plt.yticks([10*i for i in range(3, 11)], fontsize=13)
# fig.set_xlabel('Percentage (%)', fontsize=13)
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ1_DatasetB_HitRate.png")

# Dataset B: Anomaly Detection
# threshold=[(1-value)*100 for value in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]]
# f1_score=[value*100 for value in [0.65596, 0.68013, 0.70281, 0.73556, 0.76003, 0.79499, 0.8249, 0.85367, 0.90062, 0.95206, 1]]
# recall=[value*100 for value in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
# precision=[value*100 for value in [0.48805, 0.5153, 0.54179, 0.58173, 0.61295, 0.65973, 0.70198, 0.7447, 0.81921, 0.90851, 1]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(threshold,f1_score,color='blue',label='F1-score',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(threshold,recall,color='orange',label='Recall',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(threshold,precision,color='green',label='Precision',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='lower right', fontsize=13, frameon=False)
# plt.xlim([0, 100])
# plt.xticks([10*i for i in range(0, 11)], fontsize=13)
# plt.yticks(fontsize=13)
# # plt.ylim([30, 90])
# # plt.yticks([10*i for i in range(3, 10)])
# fig.set_xlabel('Percentage (%)', fontsize=13)
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ1_DatasetB_AnomalyDetection.png")




# RQ2: 
# Dataset A: Sampling Rate / Execution Time (ms)
# sampling_rate=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
# execution_time = [time/48*1000 for time in [4.11, 3.65, 3.16, 2.67, 2.26, 1.83, 1.37, 1.00, 0.69, 0.46, 0.31]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(sampling_rate,execution_time,color='blue',label='Top1',linewidth=1,alpha=1,marker='o',clip_on=False)
# legend = fig.legend()
# legend.remove()
# plt.xlim([0, 1])
# plt.xticks([0.1*i for i in range(0, 11)], fontsize=13)
# plt.ylim([0, 90])
# plt.yticks([10*i for i in range(0, 10)], fontsize=13)
# fig.set_xlabel('Sampling Rate', fontsize=13)
# fig.set_ylabel('Execution Time (ms)', fontsize=13)
# figall.savefig("RQ2_DatasetA_ExecutionTime.png")

# Dataset A: Sampling Rate / Hit Rate (%)
# sampling_rate=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
# HR_1=[rate/41*100 for rate in [18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 16]]
# HR_3=[rate/41*100 for rate in [26, 26, 26, 26, 26, 26, 25, 26, 25, 26, 26]]
# HR_5=[rate/41*100 for rate in [33, 33, 33, 33, 32, 33, 32, 32, 31, 31, 30]]
# HR_7=[rate/41*100 for rate in [35, 35, 35, 35, 34, 35, 34, 33, 33, 33, 31]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(sampling_rate,HR_1,color='blue',label='HR@1',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(sampling_rate,HR_3,color='purple',label='HR@3',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(sampling_rate,HR_5,color='orange',label='HR@5',linewidth=1,alpha=1,marker='*',clip_on=False)
# fig.plot(sampling_rate,HR_7,color='green',label='HR@7',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='lower right', bbox_to_anchor=(1, 0.21), fontsize=13, frameon=False)
# plt.xlim([0, 1])
# plt.xticks([0.1*i for i in range(0, 11)], fontsize=13)
# plt.ylim([30, 90])
# plt.yticks([10*i for i in range(3, 10)], fontsize=13)
# fig.set_xlabel('Sampling Rate', fontsize=13)
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ2_DatasetA_HitRate.png")

# Dataset B: Sampling Rate / Execution Time (ms)
# sampling_rate=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
# execution_time = [time/36*1000 for time in [1.62, 1.54, 1.24, 1.06, 0.95, 0.75, 0.66, 0.46, 0.32, 0.23, 0.20]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(sampling_rate,execution_time,color='green',label='Top1',linewidth=1,alpha=1,marker='o',clip_on=False)
# legend = fig.legend()
# legend.remove()
# plt.xlim([0, 1])
# plt.xticks([0.1*i for i in range(0, 11)], fontsize=13)
# plt.ylim([5, 45])
# plt.yticks([5*i for i in range(1, 10)], fontsize=13)
# fig.set_xlabel('Sampling Rate', fontsize=13)
# fig.set_ylabel('Execution Time (ms)', fontsize=13)
# figall.savefig("RQ2_DatasetB_ExecutionTime.png")

# Dataset B: Sampling Rate / Hit Rate (%)
# sampling_rate=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
# HR_1=[rate/25*100 for rate in [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8]]
# HR_3=[rate/25*100 for rate in [17, 17, 17, 17, 17, 17, 16, 15, 15, 15, 13]]
# HR_5=[rate/25*100 for rate in [19, 19, 19, 19, 19, 19, 19, 18, 19, 18, 17]]
# HR_7=[rate/25*100 for rate in [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 18]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(sampling_rate,HR_1,color='blue',label='HR@1',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(sampling_rate,HR_3,color='purple',label='HR@3',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(sampling_rate,HR_5,color='orange',label='HR@5',linewidth=1,alpha=1,marker='*',clip_on=False)
# fig.plot(sampling_rate,HR_7,color='green',label='HR@7',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='lower right', bbox_to_anchor=(1, 0.21), fontsize=13, frameon=False)
# plt.xlim([0, 1])
# plt.xticks([0.1*i for i in range(0, 11)], fontsize=13)
# plt.ylim([30, 90])
# plt.yticks([10*i for i in range(3, 10)], fontsize=13)
# fig.set_xlabel('Sampling Rate', fontsize=13)
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ2_DatasetB_HitRate.png")





# RQ3:
# Dataset A: Anomaly Detection
# epsilon=[0.2*i for i in range(2, 16)]
# f1_score=[value*100 for value in [0.72146469, 0.755197052, 0.764730588, 0.774937675, 0.720931166, 0.731469743, 0.703949778, 0.68589703, 0.67947323, 0.683110956, 0.688239495, 0.686434657, 0.675423259, 0.679232611]]
# recall=[value*100 for value in [1, 1, 1, 1, 0.88477, 0.87838, 0.87381, 0.84036, 0.80208, 0.78895, 0.75859, 0.72945, 0.69049, 0.68956]]
# precision=[value*100 for value in [0.56429, 0.60668, 0.61908, 0.63257, 0.60829, 0.62666, 0.58938, 0.5794, 0.58938, 0.60231, 0.62983, 0.64821, 0.661, 0.66921]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(epsilon,f1_score,color='blue',label='F1-score',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(epsilon,recall,color='orange',label='Recall',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(epsilon,precision,color='green',label='Precision',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='upper right', fontsize=13, frameon=False)
# plt.xlim([0.4, 3])
# plt.xticks([0.2*i for i in range(2, 16)], fontsize=13)
# plt.yticks(fontsize=13)
# # plt.ylim([30, 90])
# # plt.yticks([10*i for i in range(3, 10)])
# fig.set_xlabel(u'\u03B5', fontsize=13)    # epsilon
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ3_DatasetA_AnomalyDetection.png")

# Dataset A: Root Cause Localization
epsilon=[0.2*i for i in range(2, 16)]
HR_1=[rate/41*100 for rate in [16, 18, 18, 18, 12, 12, 8, 8, 8, 8, 8, 7, 8, 8]]
HR_3=[rate/41*100 for rate in [24, 26, 26, 26, 20, 20, 12, 12, 12, 12, 12, 11, 12, 12]]
HR_5=[rate/41*100 for rate in [31, 33, 33, 33, 30, 30, 17, 17, 17, 17, 17, 16, 17, 16]]
HR_7=[rate/41*100 for rate in [32, 35, 35, 35, 34, 35, 18, 18, 18, 18, 17, 17, 18, 17]]
figall=plt.figure(1)
fig = figall.add_subplot(1,1,1)
fig.grid(True)
fig.plot(epsilon,HR_1,color='blue',label='HR@1',linewidth=1,alpha=1,marker='o',clip_on=False)
fig.plot(epsilon,HR_3,color='purple',label='HR@3',linewidth=1,alpha=1,marker='x',clip_on=False)
fig.plot(epsilon,HR_5,color='orange',label='HR@5',linewidth=1,alpha=1,marker='*',clip_on=False)
fig.plot(epsilon,HR_7,color='green',label='HR@7',linewidth=1,alpha=1,marker='^',clip_on=False)
fig.legend(loc='upper right', fontsize=13, frameon=False)
plt.xlim([0.4, 3])
plt.xticks([0.2*i for i in range(2, 16)], fontsize=13)
plt.ylim([10, 90])
plt.yticks(fontsize=13)
# plt.yticks([10*i for i in range(3, 10)])
fig.set_xlabel(u'\u03B5', fontsize=13)    # epsdrilon
fig.set_ylabel('Percentage (%)', fontsize=13)
figall.savefig("RQ3_DatasetA_RootCauseLocalization.png")

# Dataset B: Anomaly Detection
# epsilon=[0.2*i for i in range(2, 16)]
# f1_score=[value*100 for value in [0.50581, 0.62592, 0.64886, 0.65596, 0.51691, 0.49622, 0.49385, 0.50603, 0.51359, 0.49005, 0.50074, 0.50936, 0.50876, 0.51346]]
# recall=[value*100 for value in [1, 1, 1, 1, 0.70401, 0.65289, 0.61433, 0.60331, 0.59565, 0.53505, 0.51454, 0.49954, 0.48454, 0.47597]]
# precision=[value*100 for value in [0.33851, 0.45552, 0.48023, 0.48805, 0.40838, 0.40019, 0.41288, 0.43577, 0.4514, 0.45203, 0.48767, 0.51958, 0.53552, 0.55735]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(epsilon,f1_score,color='blue',label='F1-score',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(epsilon,recall,color='orange',label='Recall',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(epsilon,precision,color='green',label='Precision',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='upper right', fontsize=13, frameon=False)
# plt.xlim([0.4, 3])
# plt.xticks([0.2*i for i in range(2, 16)], fontsize=13)
# plt.yticks(fontsize=13)
# # plt.ylim([30, 90])
# # plt.yticks([10*i for i in range(3, 10)])
# fig.set_xlabel(u'\u03B5', fontsize=13)    # epsilon
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ3_DatasetB_AnomalyDetection.png")

# Dataset B: Root Cause Localization
# epsilon=[0.2*i for i in range(2, 16)]
# HR_1=[rate/25*100 for rate in [9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]
# HR_3=[rate/25*100 for rate in [17, 17, 18, 17, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11]]
# HR_5=[rate/25*100 for rate in [19, 19, 20, 19, 15, 16, 14, 14, 14, 14, 14, 14, 14, 14]]
# HR_7=[rate/25*100 for rate in [20, 21, 21, 20, 17, 17, 15, 15, 15, 15, 15, 15, 15, 15]]
# figall=plt.figure(1)
# fig = figall.add_subplot(1,1,1)
# fig.grid(True)
# fig.plot(epsilon,HR_1,color='blue',label='HR@1',linewidth=1,alpha=1,marker='o',clip_on=False)
# fig.plot(epsilon,HR_3,color='purple',label='HR@3',linewidth=1,alpha=1,marker='x',clip_on=False)
# fig.plot(epsilon,HR_5,color='orange',label='HR@5',linewidth=1,alpha=1,marker='*',clip_on=False)
# fig.plot(epsilon,HR_7,color='green',label='HR@7',linewidth=1,alpha=1,marker='^',clip_on=False)
# fig.legend(loc='upper right', fontsize=13, frameon=False)
# plt.xlim([0.4, 3])
# plt.xticks([0.2*i for i in range(2, 16)], fontsize=13)
# plt.ylim([10, 90])
# plt.yticks(fontsize=13)
# # plt.yticks([10*i for i in range(3, 10)])
# fig.set_xlabel(u'\u03B5', fontsize=13)    # epsilon
# fig.set_ylabel('Percentage (%)', fontsize=13)
# figall.savefig("RQ3_DatasetB_RootCauseLocalization.png")