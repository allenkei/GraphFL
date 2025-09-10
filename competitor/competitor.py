import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import datetime
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, homogeneity_score, completeness_score, confusion_matrix



import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from CD_evaluation import evaluation, assign_predicted_clusters, dict_to_label_list, cluster_purity




seed = 42
random.seed(seed)
np.random.seed(seed)






def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--use_data', default='s1') 
  parser.add_argument('--num_node', default=120, type=int)
  parser.add_argument('--num_T', default=100, type=int)
  parser.add_argument('--num_seq', default=50, type=int)
  parser.add_argument('--data_dir', default='./data/')
  parser.add_argument('--competitor')
  parser.add_argument('-f', required=False)

  return parser.parse_args()



args = parse_args(); print(args)



###################
# LOAD SAVED DATA #
###################


print('[INFO] num_node = {}'.format(args.num_node))
data = np.load(args.data_dir +'data_{}_n{}.npz'.format(args.use_data, args.num_node))
output_dir = os.path.join("result/competitor/{}_{}_n{}".format(args.competitor, args.use_data, args.num_node))
output_cluster = pd.read_csv("result/competitor/{}_{}_n{}/output_label.csv".format(args.competitor, args.use_data, args.num_node), header=None)
T_seq = output_cluster.shape[0]



args.output_dir = output_dir
os.makedirs(output_dir, exist_ok=True)
output_holder = np.zeros((args.num_seq, 6)) # NMI, ARI, ACC, HOM, COM, PUR



for seq_iter in range(0,args.num_seq):


  true_labels = data['labels'][seq_iter]


  pred_labels = output_cluster.iloc[seq_iter]
  pred_labels = pred_labels.tolist()


  _, NMI, ARI, ACC, HOM, COM, PUR = evaluation(true_labels, pred_labels)
  
  output_holder[seq_iter, :] = [NMI, ARI, ACC, HOM, COM, PUR]




print('output_holder:\n', output_holder)
print('Mean performance:\n', np.mean(output_holder, axis=0))
print('Standard deviation of performance:\n', np.std(output_holder, axis=0))









