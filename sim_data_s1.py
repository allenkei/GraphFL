import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

data_name = "s1"
num_seq = 10
num_nodes = 100 # n
num_communities = 3
community_sizes = [40, 30, 30] 
dim_z = 5
dim_y = 15 # Number of features at each time point
hidden_dim = 64
T = 30 # Number of time points
edge_prob_intra = 0.45  # within community
edge_prob_inter = 0.15  # between communities






class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.init_rnn_weights(self.rnn)

    def forward(self, x, T):
        # x: num_nodes by dim_z
        inputs = x.unsqueeze(1).repeat(1, T, 1)  # (num_nodes, T, dim_z)
        output, _ = self.rnn(inputs)  # (num_nodes, T, hidden_dim)
        output = self.fc(output)  # (num_nodes, T, dim_y)
        return output

    def init_rnn_weights(self, rnn):
        for name, param in rnn.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.5, 0.5)
            elif 'bias' in name:
                nn.init.uniform_(param, -0.5, 0.5)








np.random.seed(42)
adj_matrices = []
labels_list = []
y_list = []
y_true_list = []





for idx in range(num_seq):

    model = RNN(input_dim=dim_z, hidden_dim=hidden_dim, output_dim=dim_y)


    labels = np.concatenate([[i] * size for i, size in enumerate(community_sizes)])
    #print("[INFO] community:", community_sizes)
    #print(f"[INFO] labels for sequence {idx}:", labels)

    means = [
        5.0 * np.ones(dim_z),        # Community 1
        1.0 * np.ones(dim_z),    # Community 2
        -5.0 * np.ones(dim_z)        # Community 3
    ]

    x = np.array([means[label] for label in labels])
    x_tensor = torch.tensor(x, dtype=torch.float32) 
    y_true = model(x_tensor, T).detach().numpy()  # num_nodes by T by dim_y


    noise = np.random.normal(0,1,y_true.shape)

    '''
    for i, label in enumerate(labels):
        if label == 0:    # Community 1
            noise[i] = np.random.normal(0, 1, size=dim_y)
        elif label == 1:  # Community 2
            noise[i] = np.random.normal(0, 1, size=dim_y)
        elif label == 2:  # Community 3
            noise[i] = np.random.normal(0, 1, size=dim_y)  
    '''
    
    y_noisy = y_true + noise

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
    y_list.append(y_noisy)
    y_true_list.append(y_true)


adj_matrices = np.array(adj_matrices)
labels_list = np.array(labels_list)
y_list = np.array(y_list)
y_true_list = np.array(y_true_list)

print("[INFO] adj_matrices.shape:",adj_matrices.shape)
print("[INFO] labels_list.shape:",labels_list.shape)
print("[INFO] y_list.shape:",y_list.shape)
print("[INFO] y_true_list.shape:",y_true_list.shape)

# Save to an .npz file
np.savez('data/data_{}.npz'.format(data_name), adj_matrices=adj_matrices, labels=labels_list, y=y_list, y_true=y_true_list)
print('[INFO] data saved')









