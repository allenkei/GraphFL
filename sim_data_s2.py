import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

data_name = "s2"
num_seq = 10
num_nodes = 150 # n
num_communities = 3
community_sizes = [50,50,50] 
dim_z = 10
dim_y = 10 # Number of features at each time point
hidden_dim = 16
T = 30 # Number of time points
edge_prob_intra = 0.35  # within community
edge_prob_inter = 0.15  # between communities




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
        nn.init.uniform_(param, -0.25, 0.25)
      elif 'bias' in name:
        nn.init.uniform_(param, -0.25, 0.25)




np.random.seed(42)
adj_matrices = []
labels_list = []
y_list = []


means = [
        -5.0 * torch.ones(dim_z), # Community 1
        0.0 * torch.ones(dim_z), # Community 2
        5.0 * torch.ones(dim_z) # Community 3
    ]

covariance_matrix = torch.eye(dim_z) * 0.1


for idx in range(num_seq):

    model = RNN(input_dim=dim_z, hidden_dim=hidden_dim, output_dim=dim_y)
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
    y_data = model(z_samples_tensor, T).detach().numpy()  # num_nodes by T by dim_y


    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
      for j in range(i + 1, num_nodes):
          if labels[i] == labels[j]:  
              prob = edge_prob_intra
          else:                       
              prob = edge_prob_inter
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

print("[INFO] adj_matrices.shape:",adj_matrices.shape)
print("[INFO] labels_list.shape:",labels_list.shape)
print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}.npz'.format(data_name), adj_matrices=adj_matrices, labels=labels_list, y=y_list)
print('[INFO] data saved')









