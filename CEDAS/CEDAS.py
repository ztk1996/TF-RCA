from __future__ import annotations

from dataclasses import dataclass, field
from random import sample
from typing import Generic, Iterable, Iterator, Optional, TypeVar

from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets

from matplotlib.colors import hsv_to_rgb
from sklearn.manifold import TSNE

# data type
T = TypeVar("T")


def distance(x: T, y: T) -> float:
    return np.linalg.norm(x - y)


# def update_center(centre: T, count: int, data_sample: T) -> T:
#     return ((count - 1) * centre + data_sample) / count


@dataclass
class MicroCluster(Generic[T]):
    centre: T
    energy: float = 1
    count: int = 1
    edges: set[MicroCluster[T]] = field(default_factory=set)
    label: str = None
    AD_selected: bool = False
    members: dict = field(default_factory=dict)    # {trace_id1: [STVector1, sample_info1], trace_id2: [STVector2, sample_info2]}

    def update_dimension(self, data_sample: T):
        # update centre
        self.centre = np.append(self.centre, [0]*(len(data_sample)-len(self.centre)))
        # update edges
        edges_list = list(self.edges)
        for idx in range(len(edges_list)):
            edges_list[idx].centre = np.append(edges_list[idx].centre, [0]*(len(data_sample)-len(edges_list[idx].centre)))
        self.edges = set(edges_list)

    def update_center(self, data_sample: T):
        self.centre = ((self.count - 1) * self.centre + data_sample) / self.count

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
    def initialization(self, data_sample: T, sample_info):
        first_sample = data_sample
        # request expert knowledge
        cluster_label = self._request_expert_knowledge(data_sample, sample_info)
        first_cluster = MicroCluster(centre=first_sample, label=cluster_label)
        first_cluster.members[sample_info['trace_id']] = [first_sample, sample_info]
        self.micro_clusters: list[MicroCluster] = [first_cluster]
        return cluster_label, 'manual'

    # 2. Update Micro-Clusters
    def Cluster_AnomalyDetector(self, data_sample: T, sample_info):
        # update Micro-Clusters dimension
        for cluster in self.micro_clusters:
            cluster.update_dimension(data_sample)
        # find nearest micro cluster
        nearest_cluster = min(
            self.micro_clusters,
            key=lambda cluster: distance(data_sample, cluster.centre),
        )
        min_dist = distance(data_sample, nearest_cluster.centre)

        if min_dist < self.r0:
            nearest_cluster.energy = 1
            nearest_cluster.count += 1
            nearest_cluster.members[sample_info['trace_id']] = [data_sample, sample_info]

            # if data is within the kernel?
            if min_dist < self.r0 / 2:
                # todo: xd
                nearest_cluster.update_center(data_sample)
            self.changed_cluster = nearest_cluster

            return nearest_cluster.label, 'auto'
        else:
            # request expert knowledge
            cluster_label = self._request_expert_knowledge(data_sample, sample_info)

            # create new micro cluster
            new_micro_cluster = MicroCluster(centre=data_sample, label=cluster_label)
            new_micro_cluster.members[sample_info['trace_id']] = [data_sample, sample_info]
            self.micro_clusters.append(new_micro_cluster)

            return new_micro_cluster.label, 'manual'
    
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
        # Check cluster label (normal, abnormal, change normal)
        while cluster_label not in ["normal", "abnormal", "change_normal"]:
            cluster_label = input("Illegal label! Please input the label of trace {}:".format(sample_info['trace_id']))
        return cluster_label

    # 3. Kill Clusters
    def kill(self) -> None:
        for i, cluster in enumerate(self.micro_clusters):
            cluster.energy -= self.decay    # 这里的衰减方式是不是得改一下，用衰减窗口

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
                <= 1.8 * self.r0    # 1.8*r0 表示两个 micro-cluster 隔多远被认为是相连的（有边存在的）
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
            elif cluster_type == 'none':
                sample_rate = 1
            sampleRates[trace_id] = sample_rate

            # if score != 1:
            #     print("find it !")
        
        self.visualization_tool()
        
        return labels, confidenceScores, sampleRates

    def visualization_tool(self):
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
                if cluster.label == 'normal':    # normal, abnormal, change_normal
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
                elif cluster.label == 'change_normal':
                    ax.add_artist(
                        plt.Circle(
                            (cluster_centers_2[idx][0], cluster_centers_2[idx][1]),
                            self.r0,
                            alpha=cluster.energy,
                            color=cluster.color,
                            clip_on=False,
                            hatch='o',    # hatch = {'/', '', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
                            linewidth=1
                        )
                    )
                idx += 1

        # plt.axis('equal')
        plt.axis('scaled')
        ax.set_xlim((np.min([center[0] for center in cluster_centers_2])-5*self.r0, np.max([center[0] for center in cluster_centers_2])+5*self.r0))
        ax.set_ylim((np.min([center[1] for center in cluster_centers_2])-5*self.r0, np.max([center[1] for center in cluster_centers_2])+5*self.r0))
        
        plt.show()
        fig.savefig("micro_clusters.png")





    def run(self) -> None:
        self.initialization()

        for data_sample in self.stream:
            self.changed_cluster = None
            self.update(data_sample)
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
