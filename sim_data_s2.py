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
  parser.add_argument('--num_seq', default=10)
  parser.add_argument('--num_nodes', default=150, type=int)
  parser.add_argument('--num_communities', default=3)
  parser.add_argument('--cov_rho', default=0.2)
  parser.add_argument('--dim_z', default=10)
  parser.add_argument('--dim_y', default=10)
  parser.add_argument('--hidden_dim', default=10)
  parser.add_argument('--T', default=30)
  parser.add_argument('--edge_prob_intra', default=0.35)
  parser.add_argument('--edge_prob_inter', default=0.15)
  
  return parser.parse_args()



args = parse_args()


if args.num_nodes == 150:
  community_sizes = [50,50,50] 
elif args.num_nodes == 300:
  community_sizes = [100,100,100] 




class RNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(RNN, self).__init__()
    self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

    self.init_rnn_weights(self.rnn)

  def forward(self, x, T):
    # x: (num_nodes, dim_z)
    inputs = x.unsqueeze(1).repeat(1, T, 1)  # (num_nodes, T, dim_z)
    output, _ = self.rnn(inputs)  # (num_nodes, T, hidden_dim)
    output = self.fc(output)  # (num_nodes, T, dim_y)
    return output


  def init_rnn_weights(self, rnn):
    for name, param in rnn.named_parameters():
      if 'weight' in name:
        nn.init.uniform_(param, -0.2, 0.2)
      elif 'bias' in name:
        nn.init.uniform_(param, -0.2, 0.2)




np.random.seed(42)
adj_matrices = []
labels_list = []
y_list = []


means = [
        -2.0 * torch.ones(args.dim_z), # Community 1
        0.0 * torch.ones(args.dim_z), # Community 2
        2.0 * torch.ones(args.dim_z) # Community 3
    ]

covariance_matrix = torch.eye(args.dim_z) * args.cov_rho


for idx in range(args.num_seq):

    model = RNN(input_dim=args.dim_z, hidden_dim=args.hidden_dim, output_dim=args.dim_y)
    model.init_rnn_weights(model.rnn)

    labels = np.concatenate([[i] * size for i, size in enumerate(community_sizes)])
    #print("[INFO] community:", community_sizes)
    #print(f"[INFO] labels for sequence {idx}:", labels)


    z_samples = []
    for i, mean in enumerate(means):
      mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
      community_samples = mvn.sample((community_sizes[i],))
      z_samples.append(community_samples)

    z_samples_tensor = torch.cat(z_samples, dim=0)
    y_data = model(z_samples_tensor, args.T).detach().numpy()  # num_nodes by T by dim_y


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









