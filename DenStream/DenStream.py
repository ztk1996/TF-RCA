import sys
import numpy as np
from sklearn import cluster
from sklearn.utils import check_array
from copy import copy
from DenStream.MicroCluster import MicroCluster
from math import ceil
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.manifold import TSNE

dbscan_label_list = []

class DenStream:

    def __init__(self, lambd=1, eps=1, beta=2, mu=2):
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
        # improvement
        self.decay = 0.001

        # self.t = 0
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize

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


    def get_labels_confidenceScores_sampleRates(self, STV_map, cluster_type):
        """
        Get labels and sample rates of traces and confidence score of each label

        Parameters
        ----------
        STV_map : {trace_id1: STVector1, trace_id2: STVector2}
        cluster_type : 'micro', 'macro', 'none'
        
        Returns
        ----------
        labels : {trace_id1: label1, trace_id2: label2}
        confidenceScores : {trace_id1: score1, trace_id2: score2}
        sampleRates : {trace_id1: rate1, trace_id2: rate2}
        """
        micro_cluster_centers = np.array([micro_cluster.center() for
                                            micro_cluster in
                                            self.p_micro_clusters + self.o_micro_clusters])
        micro_cluster_weights = [micro_cluster.weight()[0] for micro_cluster in
                                   self.p_micro_clusters + self.o_micro_clusters]
        micro_cluster_counts = [micro_cluster.count for micro_cluster in 
                                self.p_micro_clusters+self.o_micro_clusters]
        micro_cluster_scores = [np.sum(micro_cluster_counts)/micro_cluster_count
                                for micro_cluster_count in micro_cluster_counts]
        
        dbscan = DBSCAN(eps=0.3, min_samples=10, algorithm='brute')
        dbscan.fit(micro_cluster_centers, sample_weight=micro_cluster_weights)
        global dbscan_label_list
        dbscan_label_list = dbscan.labels_

        labels = {}
        confidenceScores = {}
        sampleRates = {}
        for trace_id, STVector in STV_map.items():
            STVector = np.append(STVector, [0]*(len(self.p_micro_clusters[0].center() if len(self.p_micro_clusters)!=0 
                       else self.o_micro_clusters[0].center()) - len(STVector)))
            
            nearest_index, nearest_cluster = self._get_nearest_micro_cluster(STVector, self.p_micro_clusters + self.o_micro_clusters)

            # get label 
            labels[trace_id] = nearest_cluster.label

            # get confidence score
            neighbor_index_list = [index for index, label in enumerate(dbscan.labels_) if label == dbscan.labels_[nearest_index] and label != -1]
            neighbor_cluster_list = [(self.p_micro_clusters+self.o_micro_clusters)[idx] for idx in neighbor_index_list]
            score = sum([cluster.weight() for cluster in neighbor_cluster_list if cluster.label == nearest_cluster.label]) / sum([cluster.weight() for cluster in neighbor_cluster_list]) if bool(neighbor_cluster_list) else 1
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
            elif cluster_type == 'none':
                sample_rate = 1
            sampleRates[trace_id] = sample_rate

            if score != 1:
                print("find it !")

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
            if cluster.label == 'normal':    # normal, abnormal, change_normal
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
            elif cluster.label == 'change_normal':
                ax.add_artist(
                    plt.Circle(
                        (cluster_centers_2[idx][0], cluster_centers_2[idx][1]),
                        cluster.radius(),
                        # alpha=cluster.energy,
                        color=cluster.color,
                        clip_on=False,
                        hatch='o',    # hatch = {'/', '', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
                        linewidth=1
                    )
                )

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
            current_distance = np.linalg.norm(micro_cluster.center() - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample, weight)
            if micro_cluster_copy.radius() <= self.eps:    # improvement 这里可以加上密度阈值判断，判断 count，参考 CEDAS
                micro_cluster.insert_sample(sample, weight)
                # improvement
                micro_cluster.energy = 1
                micro_cluster.count += 1
                return True
        return False

    def _merging(self, sample, sample_info, weight):
        # Update MicroCluster center dimension
        for cluster in self.p_micro_clusters + self.o_micro_clusters:
            cluster.update_center_dimension(sample)
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, weight, nearest_p_micro_cluster)
        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, weight, nearest_o_micro_cluster)
            if success:
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
                return nearest_o_micro_cluster.label, 'auto'
            else:
                # Request expert knowledge
                # improvement
                cluster_label = self._request_expert_knowledge(sample, sample_info)              

                # Create new o_micro_cluster
                micro_cluster = MicroCluster(self.lambd, sample_info['time_stamp'], cluster_label)    # improvement
                micro_cluster.insert_sample(sample, weight)
                self.o_micro_clusters.append(micro_cluster)
                return micro_cluster.label, 'manual'
        else:
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
        # Check cluster label (normal, abnormal, change normal)
        while cluster_label not in ["normal", "abnormal", "change_normal"]:
            cluster_label = input("Illegal label! Please input the label of trace {}:".format(sample_info['trace_id']))
        return cluster_label

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))

    def Cluster_AnomalyDetector(self, sample, sample_info):
        # improvement 这里各个 trace 的权重应该由已有的聚类计算出来，暂时还没想好
        sample_weight = self._validate_sample_weight(sample_weight=None, n_samples=1)
        sample_label, label_status = self._merging(sample, sample_info, sample_weight)
        # improvement 这里加上对每个簇 energy 的衰减，要不要换成时间窗衰减函数
        for cluster in self.p_micro_clusters + self.o_micro_clusters:
            cluster.energy -= self.decay

        # if sample_info["time_stamp"] % 30 == 0:
        if sample_info["time_stamp"] % self.tp == 0:    # 不懂这一步是在干啥？每隔一段时间更新所有簇的状态，有的消失，有的保留
            self.p_micro_clusters = [p_micro_cluster for p_micro_cluster
                                     in self.p_micro_clusters if
                                     p_micro_cluster.weight() >= self.beta *
                                     self.mu and p_micro_cluster.energy > 0]    # improvement 这里加上对 energy 的判断
            Xis = [((self._decay_function(sample_info["time_stamp"] - o_micro_cluster.creation_time
                                          + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in
                   self.o_micro_clusters]
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi and o_micro_cluster.energy > 0]    # improvement
        # self.t += 1
        return sample_label, label_status

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