from cProfile import label
from doctest import testfile
import random
from statistics import mean
import sys
import time
import numpy as np
from sklearn import cluster
from sklearn.utils import check_array
from copy import copy, deepcopy
from DenStream.MicroCluster import MicroCluster
from math import ceil, pow, log2
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.manifold import TSNE
from .dist_method import *
sys.path.append('..')
from db_utils import *



# 模拟人工打标的 trace（从异常纠正为正常）
traceID_list = list()
traceID_fr = open('./manual_traceID.txt', 'r')
S = traceID_fr.read()
traceID_list = [traceID for traceID in S.split(', ')]
traceID_list = random.sample(traceID_list, int(len(traceID_list)*0.00))



def ms2str(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ms/1000))
    
class DenStream:

    def __init__(self, lambd=1, eps=1, beta=2, mu=2, use_manual=False, k_std=3):
        """
        DenStream - Density-Based Clustering over an Evolving Data Stream with
        Noise.

        Parameters
        ----------
        lambd: float, optional
            The forgetting factor. The higher the value of lambda, the lower
            importance of the historical data compared to more recent data.
        eps: float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.
        beta: float, optional
            Outlier threshold. If beta ranges between 0.2 and 0.6, the clustering
            quality is very good. However, if it is set to a relatively high value 
            like 1, the quality deteriorates greatly. 
        mu: float, optional
            A smaller µ will result in a larger number of micro-clusters.

        Attributes
        ----------
        labels_ : array, shape = [n_samples]
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.

        Notes
        -----


        References
        ----------
        Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou. Density-Based
        Clustering over an Evolving Data Stream with Noise.
        """
        self.lambd = lambd
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.use_manual = use_manual
        self.k_std = k_std
        # improvement
        self.decay = 0.001

        # self.t = 0
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        # lambd
        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
            # self.tp = int(-1/self.lambd*log2(self.beta)) + 1
        else:
            self.tp = sys.maxsize
        # mu
        if self.mu == 'auto':
            self.mu = (1/(1-pow(2, -self.lambd)))

    def partial_fit(self, X, TimeStamp_list, y=None, sample_weight=None):
        """
        Online learning.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """

        #X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "data %d." % (n_features, self.coef_.shape[-1]))

        for sample, weight, time_stamp in zip(X, sample_weight, TimeStamp_list):
            self._partial_fit(sample, weight, time_stamp)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Lorem ipsum dolor sit amet

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Cluster labels
        """

        #X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "data %d." % (n_features, self.coef_.shape[-1]))

        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, weight)
        
        p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                            p_micro_cluster in
                                            self.p_micro_clusters])
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                   self.p_micro_clusters]
        dbscan = DBSCAN(eps=0.3, min_samples=10, algorithm='brute')
        dbscan.fit(p_micro_cluster_centers,
                   sample_weight=p_micro_cluster_weights)

        y = []
        for sample in X:
            index, _ = self._get_nearest_micro_cluster(sample,
                                                       self.p_micro_clusters)
            y.append(dbscan.labels_[index])

        return y
    
    
    def updateAll(self, micro_cluster):
        # if len(self.p_micro_clusters) > 0:
        #     try:
        #         mc_start = self.p_micro_clusters[0]
        #         max_update = mc_start.r_mean + mc_start.r_std*self.k_std
        #         max_w = mc_start.weight()
        #     except Exception as e:
        #         max_update = self.eps

        #     for cluster in self.p_micro_clusters:                
        #         if (cluster != micro_cluster):
        #             cluster.noNewSamples()    
        #         if cluster.weight() > max_w:
        #             max_w = cluster.weight()
        #             max_update = cluster.r_mean + cluster.r_std*self.k_std
        #     self.eps = deepcopy(max_update)
        # else:
        #     pass
        
        if len(self.p_micro_clusters) > 0:
            for cluster in self.p_micro_clusters:                
                if (cluster != micro_cluster):
                    cluster.noNewSamples()
                    # if self.use_manual == True:
                    #     db_update_weight(cluster_id=id(cluster), cluster_weight=cluster.weight()[0])

        for cluster in self.o_micro_clusters:
            if (cluster != micro_cluster):
                cluster.noNewSamples()
                # if self.use_manual == True:
                #     db_update_weight(cluster_id=id(cluster), cluster_weight=cluster.weight()[0])

    
    def get_sampleRates(self, STV_map, cluster_type):
        """
        Get sample rate of each trace

        Parameters
        ----------
        STV_map : {trace_id1: STVector1, trace_id2: STVector2}
        cluster_type : 'micro', 'macro'

        Returns
        ----------
        sampleRates : {trace_id1: rate1, trace_id2: rate2}
        """
        sampleRates = {}
        for trace_id, STVector in STV_map.items():
            STVector = np.append(STVector, [0]*(len(self.p_micro_clusters[0].center() if len(self.p_micro_clusters)!=0 
                       else self.o_micro_clusters[0].center()) - len(STVector)))

            nearest_index, nearest_cluster = self._get_nearest_micro_cluster(STVector, self.p_micro_clusters + self.o_micro_clusters)

            # get count of all micro_clusters
            clusterCounts = [cluster.count for cluster in self.p_micro_clusters+self.o_micro_clusters]
            
            # get sample rate
            # method 1
            sample_rate = 1 / (1 + np.exp(2*np.mean(clusterCounts)-nearest_cluster.count))
            # method 2
            sample_rate = nearest_cluster.count / np.sum(clusterCounts)

            sampleRates[trace_id] = sample_rate

        return sampleRates
    
    # def update_cluster_labels(self, manual_labels_list):
    #     """
    #     Update cluster labels if manual_labels_list is different. New normal traces will appear.

    #     Parameters
    #     ----------
    #     manual_labels_list : [trace_id1, trace_id2]    

    #     Returns
    #     ----------
    #     manual_labels_list : delete labels which not used in any clusters    
    #     """
    #     new_manual_labels_list = list()
    #     for manual_label in manual_labels_list:
    #         for micro_cluster in self.p_micro_clusters + self.o_micro_clusters:
    #             if manual_label in micro_cluster.members.keys():
    #                 micro_cluster.label = 'normal'
    #                 new_manual_labels_list.append(manual_label)
    #                 break
    #     return new_manual_labels_list

    def _find_centerest_edgest_members(self, micro_cluster):
        candidate_list = list()    # candidate list 中包含 7% 的元素，3% 最近的 + 4% 最远的
        if micro_cluster.label == 'normal' or micro_cluster in self.o_micro_clusters:
            return candidate_list
        distance_dict = dict()
        for trace_id, member_info in micro_cluster.members.items():
            member_STV = np.append(member_info[0], [0]*(len(micro_cluster.center())-len(member_info[0])))
            distance_dict[trace_id] = eculidDisSim(micro_cluster.center(), member_STV)
        candidate_list = sorted(distance_dict.items(), key=lambda x: x[1])[:ceil(0.03*micro_cluster.count)] + sorted(distance_dict.items(), key=lambda x: x[1])[-1*ceil(0.04*micro_cluster.count):]
        candidate_list = [item[0] for item in candidate_list]
        return candidate_list
    
    def update_cluster_and_trace_tables(self):
        cluster_items = str()
        trace_items = str()
        # get ground truth cluster labels from db
        labels_dict = db_find_cluster_labels()    # dict {cluster_id1: cluster_label1, cluster_id2: cluster_label2}
        for micro_cluster in self.p_micro_clusters + self.o_micro_clusters:
            highlight_list = self._find_centerest_edgest_members(micro_cluster)
            if id(micro_cluster) in labels_dict.keys():
                micro_cluster.label = labels_dict[str(id(micro_cluster))]
            cluster_items += "({0}, '{1}', '{2}', '{3}', {4}, {5}, {6}, {7}, {8}, {9}), ".format(id(micro_cluster), ms2str(micro_cluster.creation_time), ms2str(micro_cluster.latest_time), micro_cluster.label, micro_cluster.weight()[0], micro_cluster.svc_count_max, micro_cluster.svc_count_min, micro_cluster.rt_max, micro_cluster.rt_min, micro_cluster.avg_step)
            for trace_id in micro_cluster.members.keys():
                trace_items += "({0}, '{1}', 1), ".format(id(micro_cluster), trace_id) if trace_id in highlight_list else "({0}, '{1}', 0), ".format(id(micro_cluster), trace_id)
        # clear cluster table
        db_delete_cluster()
        # insert to cluster table
        db_insert_clusters(cluster_items=cluster_items[:-2])
        # clear trace table
        db_delete_trace()
        # insert to trace table
        db_insert_traces(trace_items=trace_items[:-2]) 
          
        
    def get_labels_confidenceScores_sampleRates(self, STV_map, cluster_type):
        """
        Get labels and sample rates of traces and confidence score of each label

        Parameters
        ----------
        STV_map : {trace_id1: STVector1, trace_id2: STVector2}
        cluster_type : 'micro', 'macro', 'none', 'rate'
        
        Returns
        ----------
        labels : {trace_id1: label1, trace_id2: label2}
        confidenceScores : {trace_id1: score1, trace_id2: score2}
        sampleRates : {trace_id1: rate1, trace_id2: rate2}
        """
        sRate = 0.8
        sampled_tid_list = list()
        if cluster_type == 'rate':
            for micro_cluster in self.p_micro_clusters + self.o_micro_clusters:
                sampled_tid_list += random.sample(micro_cluster.members.keys(), int(micro_cluster.count*sRate))


        for micro_cluster in self.p_micro_clusters + self.o_micro_clusters:
            micro_cluster.AD_selected = False

        micro_cluster_centers = np.array([micro_cluster.center() for
                                            micro_cluster in
                                            self.p_micro_clusters + self.o_micro_clusters])
        micro_cluster_weights = [micro_cluster.weight()[0] for micro_cluster in
                                   self.p_micro_clusters + self.o_micro_clusters]
        micro_cluster_counts = [micro_cluster.count for micro_cluster in 
                                self.p_micro_clusters+self.o_micro_clusters]
        micro_cluster_scores = [np.sum(micro_cluster_counts)/micro_cluster_count
                                for micro_cluster_count in micro_cluster_counts]
        
        dbscan = DBSCAN(eps=self.eps*2, min_samples=2, algorithm='brute')
        dbscan.fit(micro_cluster_centers, sample_weight=micro_cluster_weights)

        labels = {}
        confidenceScores = {}
        sampleRates = {}
        for trace_id, STVector in STV_map.items():
            STVector = np.append(STVector, [0]*(len(self.p_micro_clusters[0].center() if len(self.p_micro_clusters)!=0 
                       else self.o_micro_clusters[0].center()) - len(STVector)))
            
            nearest_index, nearest_cluster = self._get_nearest_micro_cluster(STVector, self.p_micro_clusters + self.o_micro_clusters)

            # get selected cluster
            if nearest_cluster.label == 'abnormal':
                nearest_cluster.AD_selected = True
                
            # get label 
            micro_cluster_copy = copy(nearest_cluster)
            micro_cluster_copy.insert_sample(sample=STVector, sample_info=None, weight=[1])
            if micro_cluster_copy.radius() <= 2 * self.eps:
                labels[trace_id] = nearest_cluster.label
            elif trace_id in traceID_list:
                labels[trace_id] = 'normal'
            else:
                labels[trace_id] = 'abnormal'
            labels[trace_id] = nearest_cluster.label

            # get confidence score
            # method 1
            neighbor_index_list = [index for index, label in enumerate(dbscan.labels_) if label == dbscan.labels_[nearest_index] and label != -1]
            neighbor_cluster_list = [(self.p_micro_clusters+self.o_micro_clusters)[idx] for idx in neighbor_index_list]
            score = sum([cluster.weight() for cluster in neighbor_cluster_list if cluster.label == nearest_cluster.label]) / sum([cluster.weight() for cluster in neighbor_cluster_list]) if bool(neighbor_cluster_list) else 1
            # method 2
            score = 1/nearest_cluster.count
            confidenceScores[trace_id] = score

            # get sample rate
            if cluster_type == 'micro':
                # method 1
                sample_rate = 1 / (1 + np.exp(2*np.mean(micro_cluster_scores)-np.sum(micro_cluster_counts)/nearest_cluster.count))
                # method 2
                sample_rate = (np.sum(micro_cluster_counts)/nearest_cluster.count) / np.sum(micro_cluster_scores)
                # method 3
                sample_rate = (np.sum(micro_cluster_counts)/nearest_cluster.count) / np.max(micro_cluster_scores)
            elif cluster_type == 'macro':
                neighbor_count_list = [cluster.count for cluster in neighbor_cluster_list]
                # method 1
                sample_rate = 1 / (1 + np.exp(2*np.mean(micro_cluster_scores)-((np.sum(micro_cluster_counts)/np.mean(neighbor_count_list)) if len(neighbor_count_list)!=0 else (np.sum(micro_cluster_counts)/nearest_cluster.count))))
                # method 2
                sample_rate = ((np.sum(micro_cluster_counts)/np.mean(neighbor_count_list)) if len(neighbor_count_list)!=0 else (np.sum(micro_cluster_counts)/nearest_cluster.count)) / np.sum(micro_cluster_scores)
                # method 3
                sample_rate = ((np.sum(micro_cluster_counts)/np.mean(neighbor_count_list)) if len(neighbor_count_list)!=0 else (np.sum(micro_cluster_counts)/nearest_cluster.count)) / np.max(micro_cluster_scores)
            elif cluster_type == 'rate':
                if trace_id in sampled_tid_list:
                    sample_rate = 1
                else:
                    sample_rate = 0
            elif cluster_type == 'none':
                sample_rate = 1
            sampleRates[trace_id] = sample_rate

            # if score != 1:
            #    print("find it !")

        # if len((self.p_micro_clusters+self.o_micro_clusters)) > 1:
        #     self.visualization_tool()

        return labels, confidenceScores, sampleRates
    
    def _get_same_element_index(self, ob_list, element):
        return [i for (i, v) in enumerate(ob_list) if v == element]
    
    def _get_macro_cluster(self, label_list):
        macro_clusters_list = []
        n_clusters_ = len(set(label_list))
        for label in range(-1, n_clusters_-1):
            macro_cluster_index = self._get_same_element_index(label_list, label)
            if label == -1:
                for noisy_cluster_index in macro_cluster_index:
                    macro_clusters_list.append([(self.p_micro_clusters+self.o_micro_clusters)[noisy_cluster_index]])
            else:            
                macro_cluster = [(self.p_micro_clusters+self.o_micro_clusters)[idx] for idx in macro_cluster_index]
                macro_clusters_list.append(macro_cluster)
        return macro_clusters_list

    def visualization_tool(self):
        sample_data = []
        for cluster in self.p_micro_clusters+self.o_micro_clusters:
            for data_item in cluster.members.values():
                sample_data.append(np.append(data_item[0], [0]*(len(cluster.mean)-len(data_item[0]))))    # extended STVector
        sample_data_2 = TSNE(n_components=2).fit_transform(sample_data)
        sample_data_2_trans = list(map(list, zip(*sample_data_2)))
        sample_data_x = sample_data_2_trans[0]
        sample_data_y = sample_data_2_trans[1]


        micro_cluster_centers = np.array([micro_cluster.center() for
                                          micro_cluster in
                                          self.p_micro_clusters + self.o_micro_clusters])
        dbscan = DBSCAN(eps=50, min_samples=5, algorithm='brute')
        dbscan.fit(micro_cluster_centers)
        dbscan_label_list = dbscan.labels_

        macro_clusters_list = self._get_macro_cluster(dbscan_label_list)

        for i, macro_cluster in enumerate(macro_clusters_list):
            color = hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
            for micro_cluster in macro_cluster:
                micro_cluster.color = color
        
        fig, ax = plt.subplots()

        cluster_centers = []
        for cluster in (self.p_micro_clusters+self.o_micro_clusters):
            cluster_centers.append(cluster.center())
        cluster_centers_2 = TSNE(n_components=2).fit_transform(cluster_centers)

        for idx, cluster in enumerate(self.p_micro_clusters+self.o_micro_clusters):
            if cluster.label == 'normal':    # normal, abnormal
                ax.add_artist(
                    plt.Circle(
                        (cluster_centers_2[idx][0], cluster_centers_2[idx][1]),
                        cluster.radius(),
                        # alpha=cluster.energy,
                        color=cluster.color,
                        clip_on=False,
                        hatch='*',    # hatch = {'/', '', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
                        linewidth=1
                    )
                )
            elif cluster.label == 'abnormal':
                ax.add_artist(
                    plt.Circle(
                        (cluster_centers_2[idx][0], cluster_centers_2[idx][1]),
                        cluster.radius(),
                        # alpha=cluster.energy,
                        color=cluster.color,
                        clip_on=False,
                        hatch='/',    # hatch = {'/', '', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
                        linewidth=1
                    )
                )

        plt.scatter(sample_data_x, sample_data_y, marker=".", color="black", linewidths=1)
        
        # plt.axis('equal')
        plt.axis('scaled')
        ax.set_xlim((np.min([center[0] for center in cluster_centers_2])-50, np.max([center[0] for center in cluster_centers_2])+50))
        ax.set_ylim((np.min([center[1] for center in cluster_centers_2])-50, np.max([center[1] for center in cluster_centers_2])+50))
        
        plt.show()
        fig.savefig("micro_clusters.png")
    
    def predict(self, X, y=None, sample_weight=None):
        """
        Lorem ipsum dolor sit amet

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Cluster labels
        """

        #X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)
        
        p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                            p_micro_cluster in
                                            self.p_micro_clusters])
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                   self.p_micro_clusters]
        dbscan = DBSCAN(eps=0.3, min_samples=10, algorithm='brute')
        dbscan.fit(p_micro_cluster_centers,
                   sample_weight=p_micro_cluster_weights)

        y = []
        for sample in X:
            index, _ = self._get_nearest_micro_cluster(sample,
                                                       self.p_micro_clusters)
            y.append(dbscan.labels_[index])

        return y

    def _get_nearest_micro_cluster(self, sample, micro_clusters):
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(micro_clusters):
            # current_distance = np.linalg.norm(micro_cluster.center() - sample)
            # 欧几里得相似度
            current_distance = eculidDisSim(micro_cluster.center(), sample)
            # 余弦相似度
            # current_distance = cosSim(micro_cluster.center(), sample)
            # 皮尔森相似度
            # current_distance = pearsonrSim(micro_cluster.center(), sample)
            # 曼哈顿相似度
            # current_distance = manhattanDisSim(micro_cluster.center(), sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample, sample_info, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample=sample, sample_info=sample_info, weight=weight)
            
            # all_seq = []
            # for cluster in self.p_micro_clusters+self.o_micro_clusters:
            #     for item in cluster.members.values():
            #         if item[1]['service_seq'] not in all_seq:
            #             all_seq.append(item[1]['service_seq'])
            
            # test_seq = ['start', 'ts-travel-service']
            # ['start', 'ts-travel-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-route-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-station-service', 'ts-train-service', 'ts-route-service', 'ts-price-service', 'ts-station-service', 'ts-station-service', 'ts-order-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-seat-service', 'ts-travel-service', 'ts-route-service', 'ts-order-service', 'ts-travel-service', 'ts-train-service', 'ts-config-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-basic-service', 'ts-station-service', 'ts-seat-service', 'ts-travel-service', 'ts-route-service', 'ts-order-service', 'ts-travel-service', 'ts-train-service', 'ts-config-service', 'ts-train-service']
            # ['start', 'ts-verification-code-service']
            # ['start', 'ts-order-other-service', 'ts-station-service'], 'time_seq': [950, 9]
            # ['start', 'ts-order-service', 'ts-station-service']  ['start', 'ts-order-service']  ['start', 'ts-travel-service', 'ts-ticketinfo-service', 'ts-basic-service'] ['start', 'ts-travel2-service', 'ts-ticketinfo-service', 'ts-basic-service']
            
            # ['start', 'ts-cancel-service', 'ts-order-service', 'ts-order-service', 'ts-inside-payment-service', 'ts-user-service']
            # ['start', 'ts-inside-payment-service', 'ts-order-other-service', 'ts-payment-service', 'ts-order-other-service']    ['start', 'ts-execute-service', 'ts-order-service']
            
            # if sample_info['service_seq'] == test_seq:
            #     print("find if !")

            # if sample_info['trace_bool'] == 1 and micro_cluster.label == "normal": # 越远越好 找最小 10095
            #     print("check it !")
            # elif sample_info['trace_bool'] == 1 and micro_cluster.label == "abnormal": # 越近越好 找最大 20186
            #     print("check it !")
            # elif sample_info['trace_bool'] == 0 and micro_cluster.label == 'normal': # 越近越好 找最大 1660 2197 10744
            #     print("check it !")
            # elif sample_info['trace_bool'] == 0 and micro_cluster.label == 'abnormal': # 越远越好 找最小 870  27.48 494 71.55
            #     print("check it !")
            if micro_cluster_copy.radius() <= self.eps:
            # if micro_cluster_copy.radius() <= self.eps or (len(sample_info['service_seq'])>=50 and (sample_info['service_seq'] in [item[1]['service_seq'] for item in micro_cluster.members.values()]) and micro_cluster_copy.radius()/len(sample_info['service_seq'])<=11):    # self.eps 越大则簇的个数越少，更多的样本将被归为一簇 improvement 这里可以加上密度阈值判断，判断 count，参考 CEDAS
                micro_cluster.insert_sample(sample=sample, sample_info=sample_info, weight=weight)
                # if self.use_manual == True:
                #     db_insert_trace(cluster_id=id(micro_cluster), trace_id=sample_info['trace_id'])
                #     db_update_weight(cluster_id=id(micro_cluster), cluster_weight=micro_cluster.weight()[0])
                # improvement
                # if sample_info['trace_id'] == '5198a233d36343fca02e8de831101221.41.16513363709480003':
                    # micro_cluster.label = 'normal'




                # 模拟人工打标的 trace（从异常纠正为正常）
                if sample_info['trace_id'] in traceID_list:
                    micro_cluster.label = 'normal'
                



                # micro_cluster.energy = 1
                micro_cluster.count += 1
                # update latest time
                micro_cluster.latest_time = sample_info['time_stamp']
                # update max / min svc count
                if len(sample_info['service_seq'])>micro_cluster.svc_count_max:
                    micro_cluster.svc_count_max = len(sample_info['service_seq'])
                if len(sample_info['service_seq'])<micro_cluster.svc_count_min:
                    micro_cluster.svc_count_min = len(sample_info['service_seq'])
                # update max / min rt
                if max(sample_info['time_seq'])>micro_cluster.rt_max:
                    micro_cluster.rt_max = max(sample_info['time_seq'])
                if max(sample_info['time_seq'])<micro_cluster.rt_min:
                    micro_cluster.rt_min = max(sample_info['time_seq'])
                # update avg time stamp
                micro_cluster.avg_step = (micro_cluster.latest_time - micro_cluster.creation_time)/micro_cluster.count    # ms
                # Add new member
                micro_cluster.members[sample_info['trace_id']] = [sample, sample_info]
                self.updateAll(micro_cluster)
                return True
        return False

    def _merging(self, sample, sample_info, weight, data_status, manual_labels_list):
        # 若到来的样本在人工标注字典中出现过，则所属的簇直接标记正常
        # Update MicroCluster center dimension
        for cluster in self.p_micro_clusters + self.o_micro_clusters:
            cluster.update_center_dimension(sample)
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, sample_info, weight, nearest_p_micro_cluster)
        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, sample_info, weight, nearest_o_micro_cluster)
            if success:
                # 若连续两个元素均归属一个簇，则这个簇的权重会超过阈值，从噪声簇转核心簇
                # self.beta * self.mu 越小，簇越容易从噪声转核心
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
                if sample_info['trace_id'] in traceID_list:    # 这是为 recluster 准备的
                    nearest_o_micro_cluster.label = 'normal'
                return nearest_o_micro_cluster.label, 'auto'
            else:
                # Request expert knowledge
                # improvement
                # cluster_label = self._request_expert_knowledge(sample, sample_info)
                # 人标注对其的影响也要考虑上，不一定绝对是‘abnormal’。若这个样本在人工标注字典中，则cluster_label为true
                cluster_label = 'normal' if data_status=='init' or sample_info['trace_id'] in traceID_list else 'abnormal'              

                # Create new o_micro_cluster
                micro_cluster = MicroCluster(self.lambd, sample_info['time_stamp'], cluster_label)    # improvement
                micro_cluster.insert_sample(sample=sample, sample_info=sample_info, weight=weight)
                # if self.use_manual == True:
                #     db_insert_cluster(cluster_id=id(micro_cluster), create_time=ms2str(sample_info['time_stamp']), cluster_label=cluster_label, cluster_weight=micro_cluster.weight()[0])
                #     db_insert_trace(cluster_id=id(micro_cluster), trace_id=sample_info['trace_id'])
                # Add new member
                micro_cluster.members[sample_info['trace_id']] = [sample, sample_info]
                # Update max / min svc count
                micro_cluster.svc_count_max = len(sample_info['service_seq'])
                micro_cluster.svc_count_min = len(sample_info['service_seq'])
                # Update max / min rt
                micro_cluster.rt_max = max(sample_info['time_seq'])
                micro_cluster.rt_min = max(sample_info['time_seq'])
                
                # temp
                # for clusterTest in [cluster for cluster in self.p_micro_clusters+self.o_micro_clusters if cluster.label=='normal']:
                #     if np.linalg.norm(clusterTest.center()-micro_cluster.center()) < 2 * self.eps:
                #         micro_cluster.label = 'normal'
            
                self.o_micro_clusters.append(micro_cluster)
                self.updateAll(micro_cluster)
                return micro_cluster.label, 'manual'
        else:
            if sample_info['trace_id'] in traceID_list:
                nearest_p_micro_cluster.label = 'normal'
            return nearest_p_micro_cluster.label, 'auto'

    def _request_expert_knowledge(self, sample, sample_info):
        # improvement
        # print("Trace Info:" + "\n" +
        #       "--------------------" + "\n" +
        #       "trace id: {}".format(sample_info['trace_id']) + "\n" +
        #       "trace bool: {}".format("abnormal" if sample_info['trace_bool']==1 else "normal") + "\n" +
        #       # "duration: {}".format(duration) + "\n" +
        #       # "trace structure: {}".format(structure) + "\n" +
        #       "--------------------")

        cluster_label = "abnormal" if sample_info['trace_bool']==1 else "normal" 
        # cluster_label = input("Please input the label of trace {}:".format(sample_info['trace_id']))
        # Check cluster label (normal, abnormal)
        while cluster_label not in ["normal", "abnormal"]:
            cluster_label = input("Illegal label! Please input the label of trace {}:".format(sample_info['trace_id']))
        return cluster_label

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))

    def Cluster_AnomalyDetector(self, sample, sample_info, data_status, manual_labels_list, stage=None):
        # improvement 这里各个 trace 的权重应该由已有的聚类计算出来，暂时还没想好
        sample_weight = self._validate_sample_weight(sample_weight=None, n_samples=1)
        sample_label, label_status = self._merging(sample, sample_info, sample_weight, data_status, manual_labels_list)
        # improvement 这里加上对每个簇 energy 的衰减，要不要换成时间窗衰减函数
        # for cluster in self.p_micro_clusters + self.o_micro_clusters:
        #     cluster.energy -= self.decay

        # 每隔一段时间更新所有簇的状态，有的消失，有的保留
        if sample_info["time_stamp"] % self.tp == -1:    # self.tp 越大则销毁簇越慢，保留的簇个数越多；self.tp 越小则销毁簇越快，保留簇个数越少
            # old_p_micro_clusters = self.p_micro_clusters
            self.p_micro_clusters = [p_micro_cluster for p_micro_cluster
                                     in self.p_micro_clusters if
                                     p_micro_cluster.weight() >= self.beta *
                                     self.mu]
                                     #self.mu and p_micro_cluster.energy > 0]    # improvement 这里加上对 energy 的判断
            # delete_p_micro_clusters = list(set(old_p_micro_clusters)^set(self.p_micro_clusters))
            # if len(delete_p_micro_clusters)!=0 and self.use_manual==True:
            #     for micro_cluster in delete_p_micro_clusters:
            #         db_delete_clusterid(id(micro_cluster))
            if stage == 'reCluster':
                Xis = [self.beta * self.mu for o_micro_cluster in self.o_micro_clusters]
            else:
                Xis = [((self._decay_function(sample_info["time_stamp"] - o_micro_cluster.creation_time + self.tp) - 1) /
                        (self._decay_function(self.tp) - 1)) for o_micro_cluster in self.o_micro_clusters]
            # old_o_micro_clusters = self.o_micro_clusters
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi]
                                     # and o_micro_cluster.energy > 0]    # improvement
            # delete_o_micro_clusters = list(set(old_o_micro_clusters)^set(self.o_micro_clusters))
            # if len(delete_o_micro_clusters)!=0 and self.use_manual==True:
            #     for micro_cluster in delete_o_micro_clusters:
            #         db_delete_clusterid(id(micro_cluster))
        # self.t += 1
        return sample_label, label_status

    def kill_micro_clusters(self, current_time_stamp, stage=None):
        self.p_micro_clusters = [p_micro_cluster for p_micro_cluster
                                in self.p_micro_clusters if
                                p_micro_cluster.weight() >= self.beta *
                                self.mu]
        if stage == 'reCluster':
            Xis = [self.beta * self.mu for o_micro_cluster in self.o_micro_clusters]
        else:
            Xis = [((self._decay_function(current_time_stamp - o_micro_cluster.creation_time + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in self.o_micro_clusters]
        self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                zip(Xis, self.o_micro_clusters) if
                                o_micro_cluster.weight() >= Xi]

    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                       order="C")
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of X and sample_weight do not match.")
        return sample_weight



# if __name__ == "__main__":
# data = np.random.random([1000, 5]) * 1000
# clusterer = DenStream(lambd=0.1, eps=100, beta=0.5, mu=3)
# for row in data:
#     clusterer.partial_fit([row], 1)
#     print(f"Number of p_micro_clusters is {len(clusterer.p_micro_clusters)}")
#     print(f"Number of o_micro_clusters is {len(clusterer.o_micro_clusters)}")