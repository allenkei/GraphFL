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

if args.grid_size == 14:
  args.num_nodes = args.grid_size ** 2




def generate_time_series(labels, T=100, sigma=0.25):
    """
    Parameters:
        labels (list or 1D tensor): Cluster label for each node (length n)
        T (int): Length of each time series
        sigma (float): Standard deviation of white noise

    Returns:
        torch.Tensor: (n, T) tensor of time series
    """
    labels = torch.tensor(labels, dtype=torch.long)
    n = len(labels)
    output = torch.empty((n, T))

    # Fixed cluster parameters
    A_list   = [0.5, 0.6, 0.7, 0.8]
    f_list   = [0.8, 1.0, 1.2, 1.4]
    phi_list = [0.1, 0.2, 0.3, 0.4]

    time = torch.linspace(0, 2 * torch.pi, T)

    # Precompute cluster prototype time series
    cluster_ts = []
    for A, f, phi in zip(A_list, f_list, phi_list):
        mu = A * torch.sin(f * time + phi)
        cluster_ts.append(mu)

    # Generate time series for each node based on label
    for i in range(n):
        c = labels[i]
        noise = sigma * torch.randn(T)
        output[i] = cluster_ts[c] + noise

    return output




np.random.seed(42)
adj_matrices = []
labels_list = []
y_list = []


for idx in range(args.num_seq):


    grid_size = args.grid_size
    num_nodes = grid_size * grid_size

    grid_graph = nx.grid_2d_graph(grid_size, grid_size)
    node_list = list(grid_graph.nodes())  # List of (row, col) tuples
    node_index = {node: i for i, node in enumerate(node_list)}  # Mapping (row, col) -> index

    labels = np.zeros(num_nodes, dtype=int)
    
    for i, (row, col) in enumerate(node_list):
        if row < grid_size // 2 and col < grid_size // 2:
            labels[i] = 0  # Top-left region
        elif row < grid_size // 2 and col >= grid_size // 2:
            labels[i] = 1  # Top-right region
        elif row >= grid_size // 2 and col < grid_size // 2:
            labels[i] = 2  # Bottom-left region
        else:
            labels[i] = 3  # Bottom-right region


    adj_matrix = nx.adjacency_matrix(grid_graph).toarray()
    y_data = generate_time_series(labels)  # (n, T)


    
    # Append the results for this sequence
    adj_matrices.append(adj_matrix)
    labels_list.append(labels)
    y_list.append(y_data)



data_np = y_list[0]  # shape: (n, T)
data_label = labels_list[0]
n, T = data_np.shape
colors = ['red', 'green', 'blue', 'yellow'] # 4 community

plt.figure(figsize=(12, 6))
for i in range(n):
    plt.plot(range(T), data_np[i], color=colors[data_label[i]], alpha=0.3)

plt.xlabel('Time step (T)')
plt.ylabel('Value')
plt.title(f'Line Plot of {n} Time Series')
plt.grid(True)
plt.tight_layout()
plt.savefig("data/data_{}_n{}_plot.pdf".format(args.data_name,args.grid_size**2))
plt.close()


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









