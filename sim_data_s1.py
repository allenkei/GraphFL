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
  community_sizes = [30,40,50] 
elif args.num_nodes == 210:
  community_sizes = [60,70,80] 





def generate_time_series_ar1_cluster(
    labels,
    T=100,
    rho_by_cluster=None,     
    mu_by_cluster=None,      
    sigma_by_cluster=None,   
    seed=123
):


    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)

    if rho_by_cluster is None:
        rho_by_cluster = [0.5, 0.5, 0.5, 0.5]
    if mu_by_cluster is None:
        mu_by_cluster = [0.0, 1.0, -1.0, 2.0]
    if sigma_by_cluster is None:
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

    labels = np.concatenate([[i] * size for i, size in enumerate(community_sizes)])
    #print("[INFO] community:", community_sizes)
    #print(f"[INFO] labels for sequence {idx}:", labels)


    y_data = generate_time_series_ar1_cluster(labels)  # (n, T)

    graph = nx.Graph()
    graph.add_nodes_from(range(args.num_nodes))

    for i in range(args.num_nodes):
      for j in range(i + 1, args.num_nodes):
          if labels[i] == labels[j]:  
              prob = args.edge_prob_intra
          else:                       
              prob = args.edge_prob_inter
          if np.random.binomial(1, prob):  # Sample from Bernoulli
              graph.add_edge(i, j)

    adj_matrix = nx.adjacency_matrix(graph).toarray()

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
np.savez('data/data_{}_n{}.npz'.format(args.data_name,args.num_nodes), adj_matrices=adj_matrices, labels=labels_list, y=y_list)
print('[INFO] data saved\n')









