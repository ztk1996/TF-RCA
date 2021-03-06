from __future__ import annotations

from dataclasses import dataclass, field
from random import sample
from typing import Generic, Iterable, Iterator, Optional, TypeVar

from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets

from matplotlib.colors import hsv_to_rgb
from sklearn.manifold import TSNE
import random

# data type
T = TypeVar("T")


def distance(x: T, y: T) -> float:
    return np.linalg.norm(x - y)


# def update_center(centre: T, count: int, sample: T) -> T:
#     return ((count - 1) * centre + sample) / count


@dataclass
class MicroCluster(Generic[T]):
    centre: T
    energy: float = 1
    count: int = 1
    edges: set[MicroCluster[T]] = field(default_factory=set)
    label: str = None
    AD_selected: bool = False
    members: dict = field(default_factory=dict)    # {trace_id1: [STVector1, sample_info1], trace_id2: [STVector2, sample_info2]}

    def update_dimension(self, sample: T):
        # update centre
        self.centre = np.append(self.centre, [0]*(len(sample)-len(self.centre)))
        # update edges
        edges_list = list(self.edges)
        for idx in range(len(edges_list)):
            edges_list[idx].centre = np.append(edges_list[idx].centre, [0]*(len(sample)-len(edges_list[idx].centre)))
        self.edges = set(edges_list)

    def update_center(self, sample: T):
        self.centre = ((self.count - 1) * self.centre + sample) / self.count

    def __hash__(self) -> int:
        return hash(self.energy)

    def __eq__(self, o: object) -> bool:
        xd = super().__eq__(o)
        return xd


class CEDAS(Generic[T]):
    def __init__(
        self,
        # 0. Parameter Selection
        r0: float,    # radius
        decay: float,    # a time based decay value
        threshold: int,    # expert knowledge
    ) -> None:
        self.r0 = r0
        self.decay = decay
        self.micro_clusters: MicroCluster = []
        self.changed_cluster: Optional[MicroCluster] = None
        self.threshold = threshold

    # 1. Initialization
    def initialization(self, sample: T, sample_info, data_status, manual_labels_list):
        first_sample = sample
        # request expert knowledge
        # cluster_label = self._request_expert_knowledge(sample, sample_info) 
        cluster_label = 'normal' if data_status=='init' or sample_info['trace_id'] in manual_labels_list else 'abnormal'
        first_cluster = MicroCluster(centre=first_sample, label=cluster_label)
        first_cluster.members[sample_info['trace_id']] = [first_sample, sample_info]
        self.micro_clusters: list[MicroCluster] = [first_cluster]
        return cluster_label, 'auto'

    # 2. Update Micro-Clusters
    def Cluster_AnomalyDetector(self, sample: T, sample_info, data_status, manual_labels_list):
        # update Micro-Clusters dimension
        for cluster in self.micro_clusters:
            cluster.update_dimension(sample)
        # find nearest micro cluster
        nearest_cluster = min(
            self.micro_clusters,
            key=lambda cluster: distance(sample, cluster.centre),
        )
        min_dist = distance(sample, nearest_cluster.centre)

        if min_dist < self.r0:
            nearest_cluster.energy = 1
            nearest_cluster.count += 1
            nearest_cluster.members[sample_info['trace_id']] = [sample, sample_info]

            # if data is within the kernel?
            if min_dist < self.r0 / 2:
                # todo: xd
                nearest_cluster.update_center(sample)
            self.changed_cluster = nearest_cluster

            if sample_info['trace_id'] in manual_labels_list:
                nearest_cluster.label = 'normal'
            
            return nearest_cluster.label, 'auto'
        else:
            # request expert knowledge
            # cluster_label = self._request_expert_knowledge(sample, sample_info)
            cluster_label = 'normal' if data_status=='init' or sample_info['trace_id'] in manual_labels_list else 'abnormal' 

            # create new micro cluster
            new_micro_cluster = MicroCluster(centre=sample, label=cluster_label)
            new_micro_cluster.members[sample_info['trace_id']] = [sample, sample_info]
            self.micro_clusters.append(new_micro_cluster)

            return new_micro_cluster.label, 'auto'
    
    # Request expert knowledge
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

    # 3. Kill Clusters
    def kill(self) -> None:
        for i, cluster in enumerate(self.micro_clusters):
            cluster.energy -= self.decay    # ????????????????????????????????????????????????????????????

            if cluster.energy < 0:
                # Remove all edges containing the micro-cluster
                for c in self.micro_clusters:
                    c.edges.discard(cluster)

                self.changed_cluster = cluster
                self.update_graph()

                # remove cluster
                del self.micro_clusters[i]

    # 4. Update Cluster Graph
    def update_graph(self) -> None:
        if self.changed_cluster and self.changed_cluster.count > self.threshold:
            # find neighbors
            neighbors = {
                cluster
                for cluster in self.micro_clusters
                if distance(cluster.centre, self.changed_cluster.centre)
                <= 1.8 * self.r0    # 1.8*r0 ???????????? micro-cluster ???????????????????????????????????????????????????
                and cluster.count > self.threshold
            }
            self.changed_cluster.edges = neighbors

            for cluster in neighbors:
                cluster == self.changed_cluster
                cluster.edges.add(self.changed_cluster)

            for cluster in self.micro_clusters:
                if self.changed_cluster in cluster.edges and cluster not in neighbors:
                    cluster.edges.remove(self.changed_cluster)

    def get_sampleRates(self, STV_map):
        """
        Get sample rate of each trace

        Parameters
        ----------
        STV_map : {trace_id1: STVector1, trace_id2: STVector2}

        Returns
        ----------
        sampleRates : {trace_id1: rate1, trace_id2: rate2}
        """
        sampleRates = {}
        for trace_id, STVector in STV_map.items():
            STVector = np.append(STVector, [0]*(len(self.micro_clusters[0].centre) - len(STVector)))

            nearest_cluster = min(self.micro_clusters, key=lambda cluster: distance(STVector, cluster.centre),)

            # get count of all micro_clusters
            clusterCounts = [cluster.count for cluster in self.micro_clusters]

            # get sample rate
            # method 1
            sample_rate = 1 / (1 + np.exp(2*np.mean(clusterCounts)-nearest_cluster.count))
            # method 2
            sample_rate = nearest_cluster.count / np.sum(clusterCounts)

            sampleRates[trace_id] = sample_rate

        return sampleRates

    def update_cluster_labels(self, manual_labels_list):
        """
        Update cluster labels if manual_labels_list is different. New normal traces will appear.

        Parameters
        ----------
        manual_labels_list : [trace_id1, trace_id2]    

        Returns
        ----------
        manual_labels_list : delete labels which not used in any clusters    
        """
        new_manual_labels_list = list()
        for manual_label in manual_labels_list:
            for micro_cluster in self.micro_clusters:
                if manual_label in micro_cluster.members.keys():
                    micro_cluster.label = 'normal'
                    new_manual_labels_list.append(manual_label)
                    break
        return new_manual_labels_list   
    
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
            for micro_cluster in self.micro_clusters:
                sampled_tid_list += random.sample(micro_cluster.members.keys(), int(micro_cluster.count*sRate))

        for micro_cluster in self.micro_clusters:
            micro_cluster.AD_selected = False

        labels = {}
        confidenceScores = {}
        sampleRates = {}
        micro_cluster_counts = [micro_cluster.count for micro_cluster in self.micro_clusters]
        micro_cluster_scores = [np.sum(micro_cluster_counts)/micro_cluster_count
                                for micro_cluster_count in micro_cluster_counts]
        for trace_id, STVector in STV_map.items():
            STVector = np.append(STVector, [0]*(len(self.micro_clusters[0].centre) - len(STVector)))

            nearest_cluster = min(self.micro_clusters, key=lambda cluster: distance(STVector, cluster.centre),)

            # get selected cluster
            if nearest_cluster.label == 'abnormal':
                nearest_cluster.AD_selected = True
                
            # get label
            labels[trace_id] = nearest_cluster.label

            # get confidence score
            # method 1
            score = sum([cluster.energy for cluster in nearest_cluster.edges if cluster.label == nearest_cluster.label]) / sum([cluster.energy for cluster in nearest_cluster.edges]) if bool(nearest_cluster.edges) else 1             
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
                neighbor_count_list = [cluster.count for cluster in nearest_cluster.edges]
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
            #     print("find it !")
        
        # self.visualization_tool()
        
        return labels, confidenceScores, sampleRates

    def visualization_tool(self):
        sample_data = []
        for cluster in self.micro_clusters:
            for data_item in cluster.members.values():
                sample_data.append(np.append(data_item[0], [0]*(len(cluster.centre)-len(data_item[0]))))    # extended STVector
        sample_data_2 = TSNE(n_components=2).fit_transform(sample_data)
        sample_data_2_trans = list(map(list, zip(*sample_data_2)))
        sample_data_x = sample_data_2_trans[0]
        sample_data_y = sample_data_2_trans[1]

        macro_clusters_list = self._get_macro_cluster()

        for i, macro_cluster in enumerate(macro_clusters_list):
            color = hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
            for micro_cluster in macro_cluster:
                micro_cluster.color = color
        
        fig, ax = plt.subplots()

        cluster_centers = []
        for cluster in self.micro_clusters:
            if cluster.count > self.threshold:
                cluster_centers.append(cluster.centre)
        cluster_centers_2 = TSNE(n_components=2).fit_transform(cluster_centers)

        idx = 0
        for cluster in self.micro_clusters:
            if cluster.count > self.threshold:
                if cluster.label == 'normal':    # normal, abnormal
                    ax.add_artist(
                        plt.Circle(
                            (cluster_centers_2[idx][0], cluster_centers_2[idx][1]),
                            self.r0,
                            alpha=cluster.energy,
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
                            self.r0,
                            alpha=cluster.energy,
                            color=cluster.color,
                            clip_on=False,
                            hatch='/',    # hatch = {'/', '', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
                            linewidth=1
                        )
                    )
                idx += 1

        plt.scatter(sample_data_x, sample_data_y, marker=".", color="black", linewidths=1)
        
        # plt.axis('equal')
        plt.axis('scaled')
        ax.set_xlim((np.min([center[0] for center in cluster_centers_2])-5*self.r0, np.max([center[0] for center in cluster_centers_2])+5*self.r0))
        ax.set_ylim((np.min([center[1] for center in cluster_centers_2])-5*self.r0, np.max([center[1] for center in cluster_centers_2])+5*self.r0))
        
        plt.show()
        fig.savefig("micro_clusters.png")





    def run(self) -> None:
        self.initialization()

        for sample in self.stream:
            self.changed_cluster = None
            self.update(sample)
            self.kill()

            if self.changed_cluster and self.changed_cluster.count > self.threshold:
                self.update_graph()





    def _get_macro_cluster(self) -> list[set[MicroCluster]]:
        seen: set[MicroCluster] = set()

        def dfs(cluster) -> set[MicroCluster]:
            seen.add(cluster)
            return {cluster}.union(
                *map(dfs, [edge for edge in cluster.edges if edge not in seen])
            )
        result = []
        for cluster in self.micro_clusters:
            if cluster.count > self.threshold:
                if cluster not in seen:
                    result.append(dfs(cluster))
        return result


if __name__ == "__main__":
    # data = np.genfromtxt("data.csv", delimiter=",")

    datasets = [
        {
            "data": datasets.make_circles(n_samples=1500, factor=0.5, noise=0.05),
            "xlim": [-1.5, 1.5],
            "ylim": [-1.5, 1.5],
            "r": 0.19,
        },
        {
            "data": datasets.make_moons(n_samples=1500, noise=0.05),
            "xlim": [-1.5, 2.5],
            "ylim": [-1.0, 1.5],
            "r": 0.18,
        },
        {
            "data": datasets.make_blobs(n_samples=1500, random_state=8),
            "xlim": [-10.0, 12.0],
            "ylim": [-15.0, 15.0],
            "r": 0.5,
        },
    ]

    for dataset in datasets:
        data = dataset["data"][0]

        cedas = CEDAS(
            data,
            r0=dataset["r"],
            decay=0.001,
            threshold=5,
        )
        cedas.run()

        for i, macro in enumerate(cedas.get_macro_cluster()):
            color = hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
            for cluster in macro:
                cluster.color = color

        fig = plt.figure()

        plt.scatter(data.T[0], data.T[1], marker=".", color="black")

        for cluster in cedas.micro_clusters:
            if cluster.count > cedas.threshold:
                plt.gca().add_patch(
                    plt.Circle(
                        (cluster.centre[0], cluster.centre[1]),
                        cedas.r0,
                        alpha=0.4,
                        color=cluster.color,
                    )
                )

        plt.xlim(dataset["xlim"])
        plt.ylim(dataset["ylim"])
        plt.show()
