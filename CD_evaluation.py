import torch
import numpy as np
from collections import defaultdict
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from kneed import KneeLocator
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, homogeneity_score, completeness_score, confusion_matrix


def evaluation_gamma(mu, args, seq_iter, node_degrees, adj_matrix, labels):

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



  '''
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
  '''


  # Determine the optimal number of clusters using the Elbow Method
  K_range = range(1, 11)  # Checking K from 1 to 10

  '''
  inertia = []
  for K in K_range:
      kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
      kmeans.fit(mu)
      inertia.append(kmeans.inertia_)
  '''

  silhouette_scores = []
  K_range = range(1, 11)  # Checking K from 1 to 10

  # Calculate Silhouette scores
  for K in K_range:
      if K > 1:  # Silhouette score is only valid for K > 1
          kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
          kmeans.fit(mu)
          score = silhouette_score(mu, kmeans.labels_)
          silhouette_scores.append(score)
      else:
          silhouette_scores.append(None)  # Append None for K=1

  # Filter out None values for silhouette_scores
  valid_silhouette_scores = [score for score in silhouette_scores if score is not None]
  valid_K_range = [K for K, score in zip(K_range, silhouette_scores) if score is not None]

  # Use KneeLocator to find optimal K
  # Find the maximum Silhouette Score and corresponding K
  optimal_K = valid_K_range[valid_silhouette_scores.index(max(valid_silhouette_scores))]


  # Plot the results
  plt.figure(figsize=(8, 5))
  plt.plot(K_range, silhouette_scores, marker='o', linestyle='--', label="Silhouette Score")
  plt.axvline(x=optimal_K, color='r', linestyle='--', label=f"Optimal K = {optimal_K}")
  plt.xlabel("Number of Clusters (K)")
  plt.ylabel("Silhouette Score")
  plt.title("Silhouette Score vs. Number of Clusters")
  plt.savefig( args.output_dir + '/K-mean_elbow_seq{}.png'.format(seq_iter) ) 
  plt.close()

  # Perform K-Means with optimal K
  kmeans = KMeans(n_clusters=optimal_K, random_state=42, n_init=10)
  cluster_labels = kmeans.fit_predict(mu).tolist() # [0,0,0,0,1,1,1,1]

  # want [[0,1,2,3],[4,5,6,7]]
  clusters_dict = defaultdict(list)
  for node_idx, cluster_id in enumerate(cluster_labels):
      clusters_dict[cluster_id].append(node_idx)

  clusters = list(clusters_dict.values())


  print("Optimal number of clusters:", optimal_K)
  print("Cluster assignments:", clusters)





  ###########################################
  #  FUNCTIONS DEFINED WITHIN THIS FUNCTION #
  ###########################################

  def assign_predicted_clusters(true_labels, predicted_clusters):

    true_dict = defaultdict(list)
    for idx, label in enumerate(true_labels):
        true_dict[label].append(idx)
    true_dict = dict(true_dict)

    assigned_labels = {} 
    predicted_dict = {}

    used_labels = set()
    new_label = max(true_dict.keys()) + 1  

    for i, cluster in enumerate(predicted_clusters):

        overlap_count = {label: len(set(cluster) & set(indices)) for label, indices in true_dict.items()}
        best_label = max(overlap_count, key=overlap_count.get)
        
        if overlap_count[best_label] > 0 and best_label not in used_labels:
            assigned_labels[i] = best_label
            used_labels.add(best_label)
        else:
            assigned_labels[i] = new_label
            new_label += 1 

    for i, cluster in enumerate(predicted_clusters):
        label = assigned_labels[i]
        predicted_dict[label] = cluster

    return true_dict, predicted_dict

  def dict_to_label_list(cluster_dict):

    max_index = max(max(indices) for indices in cluster_dict.values())  
    label_list = [-1] * (max_index + 1)

    for label, indices in cluster_dict.items():
        for idx in indices:
            label_list[idx] = label

    return label_list

  # label: list of label [0,0,0,1,1,1,2,2,2]
  # clusters: list of cluster by node index [[0,1,2],[3,4,5],[6,7,8]]
  true_dict, pred_dict = assign_predicted_clusters(labels, clusters) # dictionary
  true_label_list = dict_to_label_list(true_dict) # list of label
  pred_label_list = dict_to_label_list(pred_dict) # list of label






  

  # CLUSTER PURITY
  def cluster_purity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    purity = np.sum(np.amax(cm, axis=1)) / np.sum(cm)
    return purity




  ######################
  # EVALUATION METRICS #
  ######################
  print("[INFO] number of clusters:", len(clusters))
  print("[INFO] aligned labels:", pred_label_list)
  print("[INFO] true labels:", true_label_list)

  # Compute Evaluation Metrics
  NMI = normalized_mutual_info_score(true_label_list, pred_label_list)
  ARI = adjusted_rand_score(true_label_list, pred_label_list)
  ACC = accuracy_score(true_label_list, pred_label_list)
  HOM = homogeneity_score(true_label_list, pred_label_list)
  COM = completeness_score(true_label_list, pred_label_list)
  PUR = cluster_purity(true_label_list, pred_label_list)

  print("\n")
  print(f"[INFO] NMI Score: {NMI:.4f}")
  print(f"[INFO] ARI Score: {ARI:.4f}")
  print(f"[INFO] Clustering Accuracy: {ACC:.4f}")
  print(f"[INFO] Homogeneity: {HOM:.4f}")
  print(f"[INFO] Completeness: {COM:.4f}")
  print(f"[INFO] Cluster Purity: {PUR:.4f}")
  

  return clusters, NMI, ARI, ACC, HOM, COM, PUR












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




