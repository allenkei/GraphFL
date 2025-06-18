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
  
  parser.add_argument('--data_name', default="s1")
  parser.add_argument('--num_seq', default=5, type=int)
  parser.add_argument('--num_nodes', default=120, type=int)
  parser.add_argument('--num_communities', default=3)
  parser.add_argument('--dim_z', default=10)
  parser.add_argument('--T', default=100)
  parser.add_argument('--edge_prob_intra', default=0.30)
  parser.add_argument('--edge_prob_inter', default=0.15)
  
  return parser.parse_args()



args = parse_args()


if args.num_nodes == 120:
  community_sizes = [40,40,40] 
elif args.num_nodes == 240:
  community_sizes = [80,80,80] 





def generate_time_series(community_sizes, T=args.T, sigma=0.5):
    """
    Parameters:
        community_sizes (list): list of ints (number of nodes per cluster)
        T (int): Length of each time series
        sigma (float): Standard deviation of white noise

    Returns:
        torch.Tensor: Tensor of shape (n, T) with node time series
    """
    # Fixed parameters for each cluster (amplitude, frequency, phase)
    A_list   = [0.8, 1.0, 1.2]
    f_list   = [0.75, 1.0, 1.25]
    phi_list = [0.1, 0.2, 0.3]

    total_nodes = sum(community_sizes)
    time = torch.linspace(0, 2 * torch.pi, T)
    output = torch.empty((total_nodes, T))

    idx = 0
    for k, size in enumerate(community_sizes):
        mu_k = A_list[k] * torch.sin(f_list[k] * time + phi_list[k])
        noise = sigma * torch.randn((size, T))
        output[idx: (idx + size)] = mu_k + noise
        idx += size

    return output




np.random.seed(42)
adj_matrices = []
labels_list = []
y_list = []


for idx in range(args.num_seq):

    labels = np.concatenate([[i] * size for i, size in enumerate(community_sizes)])
    #print("[INFO] community:", community_sizes)
    #print(f"[INFO] labels for sequence {idx}:", labels)


    #y_data = generate_time_series(means, args.T).cpu().numpy()  # (n, T)
    y_data = generate_time_series(community_sizes)  # (n, T)

    graph = nx.Graph()
    graph.add_nodes_from(range(args.num_nodes))

    for i in range(args.num_nodes):
      for j in range(i + 1, args.num_nodes):
          if labels[i] == labels[j]:  
              prob = args.edge_prob_intra
          else:                       
              prob = args.edge_prob_inter
          if np.random.binomial(1, prob):  # Sample from Bernoulli distribution
              graph.add_edge(i, j)

    adj_matrix = nx.adjacency_matrix(graph).toarray()

    
    # Append the results for this sequence
    adj_matrices.append(adj_matrix)
    labels_list.append(labels)
    y_list.append(y_data)



data_np = y_list[0]  # shape: (n, T)
data_label = labels_list[0]
n, T = data_np.shape
colors = ['red', 'green', 'blue'] 

plt.figure(figsize=(12, 6))
for i in range(n):
    plt.plot(range(T), data_np[i], color=colors[data_label[i]], alpha=0.3)

plt.xlabel('Time step (T)')
plt.ylabel('Value')
plt.title(f'Line Plot of {n} Time Series')
plt.grid(True)
plt.tight_layout()
plt.savefig("data/data_{}_n{}_plot.pdf".format(args.data_name,args.num_nodes))
plt.close()


adj_matrices = np.array(adj_matrices)
labels_list = np.array(labels_list)
y_list = np.array(y_list)

print("[INFO] data:", args.data_name)
print("[INFO] adj_matrices.shape:",adj_matrices.shape)
print("[INFO] labels_list.shape:",labels_list.shape)
print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_n{}.npz'.format(args.data_name,args.num_nodes), adj_matrices=adj_matrices, labels=labels_list, y=y_list)
print('[INFO] data saved\n')









