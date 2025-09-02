library(reticulate)

scenario <- "s2"  ##### "s1", "s2"
num_node <- "196" ##### s1: "120", "210", s2: "144", "196"
true_K <- 4       ##### s1: 3, s2: 4

np <- import("numpy")
TS_data <- np$load(paste0("data/data_", scenario, "_n", num_node, ".npz"))
data <- TS_data$f["y"]  # seq by nodes by T
label <- TS_data$f["labels"]  # seq by nodes

T_seq <- dim(data)[1]  # Sequence length (time steps)
output_label <- array(NA, dim = dim(label))  # To store output labels

for(seq_iter in 1:T_seq){
  set.seed(42)
  
  one_data <- data[seq_iter,,]
  kmeans_result <- kmeans(one_data, centers = true_K)
  output_label[seq_iter,] <- kmeans_result$cluster
  
}


dir.create(paste0("result/c1_", scenario, "_n", num_node), showWarnings = FALSE)


write.csv(output_label, 
          paste0("result/c1_", scenario, "_n", num_node, "/output_label.csv"),
          row.names = FALSE)
