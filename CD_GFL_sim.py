import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import datetime
from CD_evaluation import evaluation_sim, data_split
torch.set_printoptions(precision=5)



seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)




def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--latent_dim', default=5, type=int)
  parser.add_argument('--output_dim', default=100, type=int)
  parser.add_argument('--num_samples', default=500)
  parser.add_argument('--langevin_K', default=50)
  parser.add_argument('--langevin_s', default=0.4) 
  parser.add_argument('--penalties', default=[0.1, 0.25, 0.5, 0.75])
  #parser.add_argument('--gamma', default=10)
  
  parser.add_argument('--epoch', default=40) # ADMM iteration
  parser.add_argument('--decoder_iteration', default=20)
  parser.add_argument('--nu_iteration', default=20)
  parser.add_argument('--decoder_lr', default=0.001)
  parser.add_argument('--decoder_thr', default=0.0001)

  parser.add_argument('--use_data', default='s1') 
  parser.add_argument('--num_node', default=150, type=int)
  parser.add_argument('--hidden_dim', default=[32,32])
  parser.add_argument('--num_seq', default=10, type=int)

  parser.add_argument('--data_dir', default='./data/')
  parser.add_argument('-f', required=False)

  return parser.parse_args()




args = parse_args(); print(args)
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print('[INFO]', device)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')




###################
# LOAD SAVED DATA #
###################


print('[INFO] num_node = {}'.format(args.num_node))
data = np.load(args.data_dir +'data_{}_n{}.npz'.format(args.use_data,args.num_node))
output_dir = os.path.join("result/{}_n{}".format(args.use_data,args.num_node))
remove_ratio = 0.1



args.output_dir = output_dir
os.makedirs(output_dir, exist_ok=True)
with open(output_dir+"/args.txt", 'w') as f:
    for arg, value in vars(args).items():
        f.write(f"{arg}: {value}\n")

def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.05, 0.05)



#########
# MODEL #
#########

class CD_temp(nn.Module):
  def __init__(self, args):
    super(CD_temp, self).__init__()

    self.d = args.latent_dim
    self.output_dim = args.output_dim
    
    self.l1 = nn.Linear(args.latent_dim, args.hidden_dim[0])
    self.l2 = nn.Linear(args.hidden_dim[0], args.hidden_dim[1])
    self.l3 = nn.Linear(args.hidden_dim[1], args.output_dim)
        
  def forward(self, z):
    # z: nm by d

    output = self.l1(z).relu()
    output = self.l2(output).relu()
    output = self.l3(output) # nm by T

    return output

    
  def infer_z(self, z, y_repeat, mu_repeat):
    # z: nm by d
    # y_repeat: nm by T (repeated by m), with y_data: n by T
    # mu_repeat: nm by d (repeated by m)

    criterion = nn.MSELoss(reduction='sum') # negative log-likelihood for Normal
    for k in range(args.langevin_K):
      z = z.detach().clone()
      z.requires_grad = True
      assert z.grad is None

      y_pred = self.forward(z) # nm by d -> nm by T
      nll = criterion(y_pred, y_repeat) # both nm by T
      z_grad_nll = torch.autograd.grad(nll, z)[0] # nm by d
      noise = torch.randn(args.num_node * args.num_samples, self.d).to(device) # nm by d

      # Langevin dynamics sampling
      z = z + torch.tensor(args.langevin_s) * (-z_grad_nll - (z - mu_repeat)) +\
          torch.sqrt(2 * torch.tensor(args.langevin_s)) * noise
          
    z = z.detach().clone() # nm by d
    return z

  
  def cal_loglik(self, mu, y):
    # mu: n by d
    # y: # n by T
    mu = mu.to(device)
    y = y.to(device)

    with torch.no_grad():

      z = mu + torch.randn_like(mu).to(device) # n by d
      output = self.l1(z).tanh()
      output = self.l2(output).tanh()
      output = self.l3(output) # n by T  
      
      log_prob = -0.5 * torch.sum((y - output)**2, dim=1)
      log_norm = -0.5 * y.shape[1] * torch.log(torch.tensor(2 * torch.pi).to(device))
      log_lik  = torch.sum(log_prob + log_norm) 
    
      return log_lik



def learn_one_seq_penalty(args, y_data, removed_y_data, removed_nodes, labels,\
    source_nodes, target_nodes, node_degrees, adj_matrix, seq_iter, pen_iter, CV):

  n = args.num_node
  m = args.num_samples
  E = args.num_edge
  d = args.latent_dim
  penalty = args.penalties[pen_iter] # lambda
  gamma = args.penalties[pen_iter] # args.gamma

  
  # initialize mu, nu, w, with zeros
  mu = torch.zeros(n, d).to(device) # zeros
  nu = torch.zeros(E, d).to(device) # zeros
  w = torch.zeros(E, d).to(device) # zeros
  
  mu_old = mu.detach().clone()
  nu_old = nu.detach().clone()


  # creat repeated version of ground truth, from (n by T) to (nm by T)
  # repeat m for n times
  y_repeat = np.repeat(y_data.numpy(), np.repeat(m, n), axis=0) # nm by T
  y_repeat = torch.from_numpy(y_repeat).to(device) # nm by T


  model = CD_temp(args).to(device)
  model.apply(init_weights)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.decoder_lr)
  criterion = nn.MSELoss(reduction='sum') # loglik sum over i, but expectation over m, so later divided by m

  for learn_iter in range(args.epoch):

    ####################
    # GENERATE SAMPLES #
    ####################
    # create repeated version of mu, from (n by d) to (nm by d)
    mu_repeat = np.repeat(mu.cpu().numpy(), np.repeat(m, n), axis=0)
    mu_repeat = torch.from_numpy(mu_repeat).to(device) # nm by d

    init_z = torch.randn(n*m, d).to(device) # nm by d, starts from N(0,1)
    sampled_z_all = model.infer_z(init_z, y_repeat, mu_repeat) # nm by d

    ##################
    # UPDATE DECODER #
    ##################
    inner_loss = float('inf')
    for decoder_iter in range(args.decoder_iteration):
      optimizer.zero_grad()
      y_pred = model(sampled_z_all) # nm by T
      loss = criterion(y_pred, y_repeat) / m 
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(),1)
      optimizer.step()

      '''
      # early stopping for decoder
      loss_relative_diff = abs( (loss.item() - inner_loss) / inner_loss )
      inner_loss = loss.item()
      if loss_relative_diff < args.decoder_thr: 
        #print('[INFO] decoder early stopping')
        break
      '''

    ################
    # UPDATE PRIOR # 
    ################
    expected_z = sampled_z_all.clone().reshape(n,m,d) # n by m by d
    expected_z = expected_z.mean(dim=1) # n by d


    mu_new = torch.zeros_like(mu)
    neighbor_sum = torch.zeros(n, d).to(device)
    

    # Calculate the terms for each edge (mu_j + nu_ij - w_ij)
    # speed up without nested for-loop
    neighbor_terms = mu[target_nodes] + nu - w # mu[target_nodes] is mu_j
    neighbor_sum.scatter_add_(0, source_nodes.unsqueeze(1).expand(-1, d), neighbor_terms)
    #print(neighbor_sum[0:15,:])

    for node_i in range(n):
      #neighbors = edge_index[1,:][edge_index[0,:] == node_i]
      mu_new[node_i] = (1.0 / (1 + gamma * node_degrees[node_i])) * (expected_z[node_i,:] + gamma * neighbor_sum[node_i,:])

    mu = mu_new.detach().clone()

    #############
    # UPDATE NU # 
    #############

    for nu_iter in range(args.nu_iteration):

      s_ij = mu[source_nodes] - mu[target_nodes] + w     # Shape: (E, d)
      s_ij_norm = torch.norm(s_ij, dim=1, keepdim=True)  # Shape: (E, 1)
      scaling_factor = 1 - (penalty / (gamma * s_ij_norm))
      scaling_factor = torch.clamp(scaling_factor, min=0.0)  # ReLU: max(0,x)
      nu = scaling_factor * s_ij
      nu = nu.detach().clone()

    ############
    # UPDATE W # 
    ############

    w = mu[source_nodes] - mu[target_nodes] - nu + w
    w = w.detach().clone()

    ############
    # RESIDUAL # 
    ############

    primal_residual = torch.sqrt(torch.mean(torch.square(mu[source_nodes] - mu[target_nodes] - nu)))
    dual_residual = torch.sqrt(torch.mean(torch.square(nu - nu_old)))

    mu_old = mu.detach().clone()
    nu_old = nu.detach().clone()

    
    if (learn_iter+1) % 10 == 0:
      print('\n\tlearning iter (seq={},[penalty={},CV={}]) ='.format(seq_iter, penalty, CV), learn_iter+1, 'of', args.epoch)
      print('\t\tprimal residual =', primal_residual)
      print('\t\tdual residual =', dual_residual)


      with torch.no_grad():
        # second row - first row
        delta_mu = torch.norm(torch.diff(mu, dim=0), p=2, dim=1)
        delta_mu = delta_mu.cpu().detach().numpy() # numpy for plot

        if CV:
          fig_name = '/CV_delta_mu_seq{}_pen{}_learn{}'.format(seq_iter,pen_iter,learn_iter+1)
        else:
          fig_name = '/FINAL_delta_mu_seq{}_pen{}_learn{}'.format(seq_iter,pen_iter,learn_iter+1)


        plt.plot(delta_mu)
        plt.savefig( output_dir + fig_name + '.png' ) 
        plt.close()
      

  if CV:
    mu_removed = mu[removed_nodes]
    log_lik = model.cal_loglik(mu_removed, removed_y_data)
    return log_lik
  else:
    torch.save(mu, os.path.join(output_dir, 'mu_par_seq{}_pen{}.pt'.format(seq_iter, pen_iter)) )
    clusters, NMI, ARI, ACC, HOM, COM, PUR= evaluation_sim(mu, args, seq_iter, node_degrees, adj_matrix, labels)

    # save here for a specific sequence and penalty
    with open(output_dir + '/clusters_seq{}_pen{}.txt'.format(seq_iter,pen_iter), 'w') as f:
      for cluster in clusters:
          f.write(' '.join(map(str, cluster)) + '\n')
    return NMI, ARI, ACC, HOM, COM, PUR
  


     

######################
# parameter learning #
######################



output_holder = torch.zeros(args.num_seq, 6) # NMI, ARI, ACC, HOM, COM, PUR


for seq_iter in range(0,args.num_seq):

  #if seq_iter == 3: break

  adj_matrix = data['adj_matrices'][seq_iter]
  labels = data['labels'][seq_iter]
  y_data = torch.tensor(data['y'][seq_iter], dtype=torch.float32) # n by T

  edge_index = np.array(np.nonzero(adj_matrix)) 
  edge_index = torch.tensor(edge_index, dtype=torch.long).to(device) # 2 by E
  source_nodes, target_nodes = edge_index.to(device) # (i,j)

  args.num_edge = edge_index.shape[1]
  node_degrees = torch.zeros(adj_matrix.shape[0], dtype=torch.float32).to(device)
  node_degrees.scatter_add_(0, edge_index[0], torch.ones(args.num_edge).to(device))

  #print('[INFO] node degree:', node_degrees.mean())

  # remove nodes
  new_y_data, removed_y_data, removed_nodes = data_split(args, y_data, edge_index, split_at=int(args.num_node/(args.num_node*remove_ratio)))
  

  print('\n')
  print('[INFO] new_y_data.shape:',new_y_data.shape)
  print('[IFNO] removed_y_data.shape:', removed_y_data.shape)
  print('[INFO] removed_nodes:', removed_nodes)
  print("[INFO] node_degrees.shape:", node_degrees.shape)
  print('[INFO] edge_index loaded with shape:', edge_index.shape)
  #print("[INFO] source_nodes.shape, target_nodes.shape:", source_nodes.shape, target_nodes.shape)

  
  output_seq = []
  for pen_iter in range(len(args.penalties)):
    loglik_seq = learn_one_seq_penalty(args, new_y_data, removed_y_data, removed_nodes, labels,\
              source_nodes, target_nodes, node_degrees, adj_matrix, 
              seq_iter, pen_iter=pen_iter, CV=True)
    output_seq.append(loglik_seq.item())
    print('[INFO] log_lik:', output_seq)

  best_index = output_seq.index(max(output_seq))
  print("\nbest_index:" , best_index)
  print(output_seq)

  # use full data, None for removed_y_data and removed_nodes 
  NMI, ARI, ACC, HOM, COM, PUR = learn_one_seq_penalty(args, y_data, None, None, labels,\
              source_nodes, target_nodes, node_degrees, adj_matrix, 
              seq_iter, pen_iter=best_index, CV=False)


  output_holder[seq_iter,:] = torch.tensor((NMI, ARI, ACC, HOM, COM, PUR))
  print('[INFO] result for seq_iter = {}:\n'.format(seq_iter), output_holder[seq_iter,:])
  torch.save(output_holder, os.path.join(output_dir, 'result_table_seq{}.pt'.format(seq_iter)) ) # have previous results



print('output_holder:\n', output_holder)
columns = ['NMI', 'ARI', 'ACC', 'HOM', 'COM', 'PUR']
df = pd.DataFrame(output_holder.cpu().numpy(), columns=columns)
df.to_csv(os.path.join(output_dir, 'output_metrics.csv'), index=False)
#torch.save(output_holder, os.path.join(output_dir, 'result_table_all.pt') ) # save all results in the last

# print result for table
print('mean perforamnce:\n', np.mean(output_holder.cpu().numpy(),axis=0))
print('std perforamnce:\n', np.std(output_holder.cpu().numpy(),axis=0)) 




