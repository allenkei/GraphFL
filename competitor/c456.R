install.packages("NAC")
library(NAC)
library(reticulate)


# load data
np <- import("numpy")
TS_data <- np$load("data/data_s4_n144.npz")
T_seq <- 10

num_node <- "144" # "120", "144"
scenario <- "s4"  # "s1", "s2"

#TS_data$files # get names
data <- TS_data$f["y"]  # seq by N by T
label <- TS_data$f["labels"] # seq by N
adj <- TS_data$f["adj_matrices"] # seq by N by N

T_seq <- dim(data)[1] # seq
seq_iter <- 1

for(seq_iter in q:T_seq){
  

  one_adj <- adj[seq_iter,,]
  one_data <- data[seq_iter,,]
  
  
  
  pred_label <- CAclustering(one_adj, one_data, 4)
  pred_label <- NAC(one_adj, one_data, 4)
  pred_label <- SDP(one_adj, one_data, lambda = 0.2, K = 3, alpha = 0.5, rho = 2, TT = 100, tol = 5)
  
  # pred_label
  # label[1,]
  
}

View(matrix(label[1,], nrow=12, ncol=12, byrow = TRUE) )
View(matrix(pred_label, nrow=12, ncol=12, byrow = TRUE) )

