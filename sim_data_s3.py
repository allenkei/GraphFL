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
  
  parser.add_argument('--data_name', default="s3")
  parser.add_argument('--num_seq', default=10)
  parser.add_argument('--grid_size', default=15, type=int)
  parser.add_argument('--cov_rho', default=0.1)
  parser.add_argument('--dim_z', default=10)
  parser.add_argument('--dim_y', default=10)
  parser.add_argument('--hidden_dim', default=10)
  parser.add_argument('--T', default=30)
  
  return parser.parse_args()



args = parse_args()




# Define an RNN model
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
        nn.init.uniform_(param, -0.3, 0.3)
      elif 'bias' in name:
        nn.init.uniform_(param, -0.3, 0.3)



adj_matrices = []
labels_list = []
y_list = []



for idx in range(args.num_seq):

    grid_size = args.grid_size
    num_nodes = grid_size * grid_size

    grid_graph = nx.grid_2d_graph(grid_size, grid_size)
    node_list = list(grid_graph.nodes())  # List of (row, col) tuples
    node_index = {node: i for i, node in enumerate(node_list)}  # Mapping (row, col) -> index

    # Define four community labels based on the grid location
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

    means = [
        -2.0 * torch.ones(args.dim_z),  # Top-left region
        -1.0 * torch.ones(args.dim_z),   # Top-right region
        0.0 * torch.ones(args.dim_z),    # Bottom-left region
        1.0 * torch.ones(args.dim_z),    # Bottom-right region
    ]
    



    '''
    for i, (row, col) in enumerate(node_list):
      if row < grid_size // 3:
          labels[i] = 0  # Top
      elif row < 2 * grid_size // 3:
          labels[i] = 1  # Middle
      else:
          labels[i] = 2  # Bottom

    means = [
        -5.0 * torch.ones(args.dim_z),  # Region 0: Top
        0.0 * torch.ones(args.dim_z),        # Region 1: Middle
        5.0 * torch.ones(args.dim_z),   # Region 2: Bottom
    ]
    '''

    covariance_matrix = torch.eye(args.dim_z) * args.cov_rho

    z_samples = torch.zeros((num_nodes, args.dim_z))
    for region in range(4):
        region_indices = np.where(labels == region)[0]
        mvn = torch.distributions.MultivariateNormal(means[region], covariance_matrix)
        z_samples[region_indices] = mvn.sample((len(region_indices),))


    rnn_model = RNN(input_dim=args.dim_z, hidden_dim=args.hidden_dim, output_dim=args.dim_y)
    rnn_model.init_rnn_weights(rnn_model.rnn)

    y_data = rnn_model(z_samples, args.T).detach().numpy() 
    adj_matrix = nx.adjacency_matrix(grid_graph).toarray()

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
np.savez('data/data_{}_n{}.npz'.format(args.data_name,args.grid_size**2), adj_matrices=adj_matrices, labels=labels_list, y=y_list)
print('[INFO] data saved')










