import torch
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score


def evaluation_gamma(mu, args, pen_iter, node_degrees, adj_matrix, labels):

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
  threshold = st.gamma.ppf(1 - alpha/(n-1), a = num_sample*args.latent_dim/2, scale = 2/num_sample)

  clusters = [] 
  clusters_mu_list = []  # list of list of mu (nested)

  for i in range(n):

    node_i_mu = mu[i, :]
    assigned_to_cluster = False

    if not clusters:
        clusters.append([i])
        clusters_mu_list.append([node_i_mu])
        assigned_to_cluster = True
    else:    
      for cluster_idx, cluster_mus in enumerate(clusters_mu_list):

        centroid = np.mean(np.stack(cluster_mus), axis=0)
        mu_diff = node_i_mu - centroid

        samples = np.random.multivariate_normal(mu_diff, 2 * np.eye(d), num_sample)
        sampled_z_norm2 = 0.5 * np.sum(samples ** 2, axis=1)
        mean_sampled_norm = np.mean(sampled_z_norm2)

        if mean_sampled_norm < threshold:
          clusters[cluster_idx].append(i)
          clusters_mu_list[cluster_idx].append(node_i_mu)
          assigned_to_cluster = True
          break

    # If not assigned to any cluster, create a new cluster
    if not assigned_to_cluster:
      clusters.append([i])
      clusters_mu_list.append([node_i_mu])


  print("[INFO] clusters:", clusters)

  #####################
  # CLUSTER ALIGNMENT #
  #####################
  # BEGIN
  def match_clusters_by_overlap(true_labels, pred_clusters):

    unique_true = np.unique(true_labels) # true cluster labels i.e. [0,1,2]
    true_clusters = {label: set(np.where(true_labels == label)[0]) for label in unique_true}

    pred_labels = np.empty_like(true_labels)
    pred_clusters_dict = {i: set(cluster) for i, cluster in enumerate(pred_clusters)}

    mapping = {}
    used_true_clusters = set()

    for pred_id, pred_nodes in pred_clusters_dict.items():
      # matching by max the overlaping
      best_match = max(true_clusters, key=lambda true_id: len(pred_nodes & true_clusters[true_id]))
      mapping[pred_id] = best_match
      used_true_clusters.add(mapping[pred_id])

    # Relabel predicted clusters according to mapping
    for pred_id, pred_nodes in pred_clusters_dict.items():
      for node in pred_nodes:
        pred_labels[node] = mapping[pred_id]

    return pred_labels
  # END



  '''
  # Modularity score
  Q = 0.0
  for community in clusters:
    for i in community:
      for j in community:
        if i != j:
          A_ij = adj_matrix[i, j]
          k_i = node_degrees[i]
          k_j = node_degrees[j]
          Q += A_ij - (k_i * k_j) / (2 * E)
  Q /= (2 * E)
  '''





  # Align predicted labels
  aligned_pred_labels = match_clusters_by_overlap(labels, clusters)

  # Compute Evaluation Metrics
  NMI = normalized_mutual_info_score(labels, aligned_pred_labels)
  ARI = adjusted_rand_score(labels, aligned_pred_labels)
  ACC = accuracy_score(labels, aligned_pred_labels)

  print(f"[INFO] NMI Score: {NMI:.4f}")
  print(f"[INFO] ARI Score: {ARI:.4f}")
  print(f"[INFO] Clustering Accuracy: {ACC:.4f}")

  return clusters, NMI, ARI, ACC












def data_split(args, y_data, edge_index, split_at):
    new_y_data = []      # For training
    removed_y_data = []  # For testing
    removed_nodes = []
  
    for node in range(args.num_node):
        if node % split_at == 0: 
            removed_y_data.append(y_data[node])
            removed_nodes.append(node)

            neighbors = edge_index[1][edge_index[0] == node].tolist() 
            
            if neighbors:
                neighbor_values = y_data[neighbors] 
                average_neighbor_value = neighbor_values.mean(dim=0)
                new_y_data.append(average_neighbor_value)
            else:
                new_y_data.append(y_data[node])
        else:
            new_y_data.append(y_data[node])
                  
    new_y_data = torch.stack(new_y_data)
    removed_y_data = torch.stack(removed_y_data)
  
    return new_y_data, removed_y_data, removed_nodes




