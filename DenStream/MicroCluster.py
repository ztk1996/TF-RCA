import numpy as np


class MicroCluster:
    def __init__(self, lambd, creation_time, micro_cluster_label):
        self.lambd = lambd
        self.decay_factor = 2 ** (-lambd)    # 此时 2**(-lambd*t) 中的 t 为 1，于是 decay_factor 为 2**(-lambd)
        self.mean = 0
        self.variance = 0
        self.sum_of_weights = 0
        self.creation_time = creation_time
        self.latest_time = creation_time
        self.svc_count_max = 0
        self.svc_count_min = 0
        self.rt_max = 0
        self.rt_min = 0
        
        # improvement
        self.label = micro_cluster_label
        self.energy = 1    # float
        self.count = 1    # int
        self.AD_selected = False
        self.members = {}    # {trace_id1: [STVector1, sample_info1], trace_id2: [STVector2, sample_info2]}
        self.color = None
        
        # self.LS = 0
        # self.SS = 0
        # self.rLS = 0
        # self.rSS = 0
        # self.r_std = 0
        # self.r_mean = 0
        # self.M2 = 0



    def insert_sample(self, sample, sample_info, weight):
        if self.sum_of_weights != 0:
            # Update sum of weights
            old_sum_of_weights = self.sum_of_weights
            new_sum_of_weights = old_sum_of_weights * self.decay_factor + weight
            
            # Update real time LS and SS
            # self.LS = np.multiply(self.LS, self.decay_factor)
            # self.SS = np.multiply(self.SS, self.decay_factor)
            # self.LS = self.LS + sample
            # self.SS = self.SS + np.power(sample, 2)

            # Update incremental radius treshold
            # n1 = self.count
            # n = self.count + 1
            # delta = self.radius() - self.r_mean
            # delta_n = delta / n
            # term1 = delta*delta_n*n1
            # self.r_mean += delta_n
            # self.M2 = self.M2 + term1 
            # self.r_std = np.sqrt((self.M2)/(n-1))

            # Update mean
            if len(sample) != len(self.mean):
                print("error !")
            old_mean = self.mean
            new_mean = old_mean + \
                (weight / new_sum_of_weights) * (sample - old_mean)

            # Update variance
            old_variance = self.variance
            new_variance = old_variance * ((new_sum_of_weights - weight)
                                           / old_sum_of_weights) \
                + weight * (sample - new_mean) * (sample - old_mean)

            self.mean = new_mean
            self.variance = new_variance
            self.sum_of_weights = new_sum_of_weights
        else:
            self.mean = sample
            self.LS = sample
            self.SS = sample
            self.variance = np.array([0]*len(sample))
            self.sum_of_weights = weight

    def update_center_dimension(self, sample):
        self.mean = np.append(self.mean, [0]*(len(sample)-len(self.mean)))
        self.variance = np.append(self.variance, [0]*(len(sample)-len(self.variance)))
        # self.LS = np.append(self.LS, [0]*(len(sample)-len(self.LS)))
        # self.SS = np.append(self.SS, [0]*(len(sample)-len(self.SS)))

    def radius(self):
        # method 1
        if self.sum_of_weights > 0:
            return np.linalg.norm(np.sqrt(self.variance / self.sum_of_weights))
        else:
            return float('nan')
        # method 2
        # LSd = np.power(np.divide(self.LS, float(self.sum_of_weights)), 2)
        # SSd = np.divide(self.SS, float(self.sum_of_weights))
        # return np.nanmax(np.sqrt((SSd.astype(float)-LSd.astype(float)))) 

    def center(self):
        # method 1
        return self.mean
        # method 2
        # return np.divide(self.LS, float(self.sum_of_weights))

    def weight(self):
        return self.sum_of_weights
    
    def noNewSamples(self):
        """
        Updates the `Weighted Linear Sum` (WLS), the `Weighted Squared Sum` (WSS) and the weight of the micro-cluster when no new samples are merged.
        """
        self.LS = np.multiply(self.LS, self.decay_factor)
        self.SS = np.multiply(self.SS, self.decay_factor)
        self.sum_of_weights *= self.decay_factor

    def __copy__(self):
        new_micro_cluster = MicroCluster(self.lambd, self.creation_time, self.label)
        new_micro_cluster.sum_of_weights = self.sum_of_weights
        new_micro_cluster.variance = self.variance
        new_micro_cluster.mean = self.mean
        new_micro_cluster.latest_time = self.latest_time
        new_micro_cluster.svc_count_max = self.svc_count_max
        new_micro_cluster.svc_count_min = self.svc_count_min
        new_micro_cluster.rt_max = self.rt_max
        new_micro_cluster.rt_min = self.rt_min
        # improvement
        new_micro_cluster.energy = self.energy    # float
        new_micro_cluster.count = self.count    # int
        new_micro_cluster.AD_selected = self.AD_selected
        new_micro_cluster.members = self.members
        new_micro_cluster.color = self.color

        # new_micro_cluster.LS = self.LS
        # new_micro_cluster.SS = self.SS
        # new_micro_cluster.rLS = self.rLS
        # new_micro_cluster.rSS = self.rSS
        # new_micro_cluster.r_std = self.r_std
        # new_micro_cluster.r_mean = self.r_mean
        # new_micro_cluster.M2 = self.M2

        return new_micro_cluster