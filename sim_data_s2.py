import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import random


seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



def parse_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--data_name', default="s2")
  parser.add_argument('--num_seq', default=5, type=int)
  parser.add_argument('--grid_size', default=14, type=int)
  parser.add_argument('--T', default=100)
  
  return parser.parse_args()



args = parse_args()

if args.grid_size == 12:
    args.num_nodes = args.grid_size ** 2
    # rows, cols are half-open intervals: [start, end)
    blocks = [
        ((0, 6),  (0, 5)),   # block 0: rows, cols
        ((6, 12), (0, 7)),   # block 1: rows, cols
        ((0, 7),  (5, 12)),  # block 2: rows, cols
        ((7, 12), (7, 12)),  # block 3: rows, cols
    ]
elif args.grid_size == 14:
    args.num_nodes = args.grid_size ** 2
    blocks = [
        ((0, 8),  (0, 5)),   # block 0
        ((8, 14), (0, 7)),   # block 1
        ((0, 9),  (5, 14)),  # block 2
        ((9, 14), (7, 14)),  # block 3
    ]





def generate_time_series_ar1_cluster(labels,seed,T=100):

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)

    rho_by_cluster = [0.5, 0.5, 0.5, 0.5]
    mu_by_cluster = [-0.8, 0.0, 0.8, 1.6]
    sigma_by_cluster = [1.0, 1.0, 1.0, 1.0]

    out = torch.empty((n, T), dtype=torch.float32)
    for i in range(n):
        g = labels[i]
        rho   = float(rho_by_cluster[g])
        mu    = float(mu_by_cluster[g])
        sigma = float(sigma_by_cluster[g])

        var0 = sigma**2 / max(1e-8, (1.0 - rho**2))
        x0 = rng.normal(mu, np.sqrt(var0))
        eps = rng.normal(0.0, sigma, size=T).astype(np.float32)

        x = np.empty(T, dtype=np.float32)
        x[0] = x0
        for t in range(1, T):
            x[t] = mu + rho * (x[t-1] - mu) + eps[t]
        out[i] = torch.from_numpy(x)

    return out






np.random.seed(42)
adj_matrices = []
labels_list = []
y_list = []


for idx in range(args.num_seq):

    grid_size = args.grid_size
    num_nodes = grid_size * grid_size

    grid_graph = nx.grid_2d_graph(grid_size, grid_size)
    node_list = list(grid_graph.nodes())  # List of (row, col) tuples
    node_index = {node: i for i, node in enumerate(node_list)}  # Mapping (row, col) to index

    labels = np.full(num_nodes, -1, dtype=int)
    for i, (row, col) in enumerate(node_list):
        for g, ((r0, r1), (c0, c1)) in enumerate(blocks):
            if r0 <= row < r1 and c0 <= col < c1:
                labels[i] = g
                break


    adj_matrix = nx.adjacency_matrix(grid_graph).toarray()
    y_data = generate_time_series_ar1_cluster(labels, seed = 42+idx*2)  # (n, T)


    adj_matrices.append(adj_matrix)
    labels_list.append(labels)
    y_list.append(y_data)



adj_matrices = np.array(adj_matrices)
labels_list = np.array(labels_list)
y_list = np.array(y_list)

print("[INFO] data:", args.data_name)
print("[INFO] adj_matrices.shape:",adj_matrices.shape)
print("[INFO] labels_list.shape:",labels_list.shape)
print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_n{}.npz'.format(args.data_name,args.grid_size**2), adj_matrices=adj_matrices, labels=labels_list, y=y_list)
print('[INFO] data saved\n')









