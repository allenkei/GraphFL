import torch
import numpy as np
from math import comb
import scipy.stats as st
import matplotlib.pyplot as plt

def evaluation_gamma(mu, args, pen_iter, node_degrees, adj_matrix):

  # mu: n by d

  torch.manual_seed(1)
  np.random.seed(1)

  alpha = 0.01 # for threshold
  n = args.num_node
  E = args.num_edge
  d = args.latent_dim
  num_sample = args.gamma_num_samples
  num_T = args.num_T
  
  mu = mu.cpu().numpy() # n by d

  # threshold from gamma distribution 
  # alpha/(num_nodes-1)
  threshold = st.gamma.ppf(1 - alpha/(n-1), a = num_sample*args.latent_dim/2, scale = 2/num_sample)

  clusters = []  # List to store clusters, where each cluster is a list of node indices
  clusters_mu_list = []  # list of list of mu (nested)

  for i in range(n):

    #print("\n")
    #print("[INFO] node:", i)
    #print("[INFO] number of cluster:", len(clusters))
    node_i_mu = mu[i, :]
    assigned_to_cluster = False

    for cluster_idx, cluster_mus in enumerate(clusters_mu_list):

      #print("[INFO] cluster_idx, centroid:", cluster_idx, centroid)
      random_node_mu = cluster_mus[np.random.choice(len(cluster_mus))]
      mu_diff = node_i_mu - random_node_mu

      samples = np.random.multivariate_normal(mu_diff, 2 * np.eye(d), num_sample)
      sampled_z_norm2 = 0.5 * np.sum(samples ** 2, axis=1)

      mean_sampled_norm = np.mean(sampled_z_norm2)
      #print("[INFO] mean_sampled_norm, threshold:", mean_sampled_norm, threshold)

      if mean_sampled_norm < threshold:
        clusters[cluster_idx].append(i)
        clusters_mu_list[cluster_idx].append(node_i_mu)
        assigned_to_cluster = True
        break

    # If not assigned to any cluster, create a new cluster
    if not assigned_to_cluster:
      clusters.append([i])  # New cluster with the current node
      clusters_mu_list.append([node_i_mu])

  

  # Modularity score (Q)
  Q = 0.0

  # For each community, calculate the contribution to modularity
  for community in clusters:
      # For each pair of nodes in the community
      for i in community:
          for j in community:
              if i != j:  # Don't double-count the diagonal
                  A_ij = adj_matrix[i, j]  # 1 if edge exists, 0 otherwise
                  k_i = node_degrees[i]  # Degree of node i
                  k_j = node_degrees[j]  # Degree of node j

                  # Modularity contribution from the pair (i, j)
                  Q += A_ij - (k_i * k_j) / (2 * E)

  # Normalize by total number of edges
  Q /= (2 * E)



  print("[INFO] clusters:", clusters)
  return clusters, Q









