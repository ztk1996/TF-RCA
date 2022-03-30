import torch
import json
import argparse
import numpy as np
import random
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import time

from DataPreprocess.STVProcess import embedding_to_vector, load_dataset, process_one_trace
from DenStream.DenStream import DenStream

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # ========================================
    # Init path vector encoder
    # ========================================
    all_path = []

    # ========================================
    # Load dataset
    # ========================================
    dataloader = load_dataset()    # trace list

    # ========================================
    # Create cluster object
    # ========================================
    denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11)


    print('Start !')
    for index, data in tqdm(enumerate(dataloader), desc="All Samples: "):
        # ========================================
        # Path vector encoder
        # ========================================
        all_path = process_one_trace(data, all_path)
        STVector = embedding_to_vector(data, all_path)

        sample_label = denstream.Cluster_AnomalyDetector(np.array(STVector), data)

                
        
    print("Done !")

if __name__ == '__main__':
    main()