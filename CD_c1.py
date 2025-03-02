import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import datetime
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from CD_evaluation import assign_predicted_clusters, dict_to_label_list, cluster_purity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, homogeneity_score, completeness_score, confusion_matrix





seed = 42
random.seed(seed)
np.random.seed(seed)





def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--output_dim', default=10, type=int)
  parser.add_argument('--use_data', default='s1') 
  parser.add_argument('--num_node', default=150, type=int)
  parser.add_argument('--num_T', default=30, type=int) # number of time points
  parser.add_argument('--num_seq', default=10)
  parser.add_argument('--data_dir', default='./data/')
  parser.add_argument('-f', required=False)

  return parser.parse_args()



args = parse_args(); print(args)


###################
# LOAD SAVED DATA #
###################

if args.use_data == 's1':
  print('[INFO] num_node = {}'.format(args.num_node))
  data = np.load(args.data_dir +'data_s1.npz')
  output_dir = os.path.join(f"result/c1_s1")
elif args.use_data == 's2':
  print('[INFO] num_node = {}'.format(args.num_node))
  data = np.load(args.data_dir +'data_s2_n{}.npz'.format(args.num_node))
  output_dir = os.path.join("result/c1_s2_n{}".format(args.num_node))
  remove_ratio = 0.1
elif args.use_data == 's3':
  print('[INFO] num_node = {}'.format(args.num_node))
  data = np.load(args.data_dir +'data_s3_n{}.npz'.format(args.num_node))
  output_dir = os.path.join("result/c1_s3_n{}".format(args.num_node))
  remove_ratio = 0.1


args.output_dir = output_dir
os.makedirs(output_dir, exist_ok=True)





###########
# K-MEANS #
###########




output_holder = np.zeros((args.num_seq, 6)) # NMI, ARI, ACC, HOM, COM, PUR




for seq_iter in range(0,args.num_seq):


  labels = data['labels'][seq_iter]
  y_data = data['y'][seq_iter] # n by T by p
  y_data = y_data.reshape(args.num_node, args.num_T * args.output_dim)  # n by T*p


  silhouette_scores = []
  K_range = range(1, 11)  # Checking K from 1 to 10

  # Calculate Silhouette scores
  for K in K_range:
      if K > 1:  # Silhouette score is only valid for K > 1
          kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
          kmeans.fit(y_data)
          score = silhouette_score(y_data, kmeans.labels_)
          silhouette_scores.append(score)
      else:
          silhouette_scores.append(None)  # Append None for K=1

  # Filter out None values for silhouette_scores
  valid_silhouette_scores = [score for score in silhouette_scores if score is not None]
  valid_K_range = [K for K, score in zip(K_range, silhouette_scores) if score is not None]

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
  cluster_labels = kmeans.fit_predict(y_data).tolist() # [0,0,0,0,1,1,1,1]


  # want [[0,1,2,3],[4,5,6,7]]
  clusters_dict = defaultdict(list)
  for node_idx, cluster_id in enumerate(cluster_labels):
      clusters_dict[cluster_id].append(node_idx)

  clusters = list(clusters_dict.values())


  print("Optimal number of clusters:", optimal_K)
  print("Cluster assignments:", clusters)


  # label: list of label [0,0,0,1,1,1,2,2,2]
  # clusters: list of cluster by node index [[0,1,2],[3,4,5],[6,7,8]]
  true_dict, pred_dict = assign_predicted_clusters(labels, clusters) # dictionary
  true_label_list = dict_to_label_list(true_dict) # list of label
  pred_label_list = dict_to_label_list(pred_dict) # list of label


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
  
  # Store the metrics for the current sequence in the output_holder
  output_holder[seq_iter, :] = [NMI, ARI, ACC, HOM, COM, PUR]




print('output_holder:\n', output_holder)
print('Mean performance:\n', np.mean(output_holder, axis=0))
print('Standard deviation of performance:\n', np.std(output_holder, axis=0))



# Convert output_holder to a DataFrame for easier handling
df = pd.DataFrame(output_holder, columns=['NMI', 'ARI', 'ACC', 'HOM', 'COM', 'PUR'])

# Save to a CSV file
output_file = os.path.join(output_dir, 'result_c1.csv')
df.to_csv(output_file, index=False)






