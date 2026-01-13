setwd("/Users/xichen/Library/CloudStorage/OneDrive-Personal/ucsc phd/Research/github/GraphFL")

library(reticulate)
library(dtwclust)


scenario <- "s2"  ##### "s1", "s2"
num_node <- "196" ##### s1: "120", "210", s2: "144", "196"
true_K <- 4       ##### s1: 3, s2: 4


np <- import("numpy")
TS_data <- np$load(paste0("data/data_",scenario,"_n",num_node,".npz"))
data  <- TS_data$f["y"]  # seq by nodes by T
label <- TS_data$f["labels"] # seq by nodes



T_seq <- dim(data)[1]
output_label <- array(NA, dim = dim(label)) # saved output labels



for(seq_iter in 1:T_seq){
  
  one_data <- data[seq_iter,,]
  
  result <- tsclust(one_data, 
                    k = true_K,
                    type = "partitional",
                    distance = "L2",
                    seed = 42)
  
  # selecting optimal K
  # result_Sil <- sapply(result, cvi, type = "Sil")
  # optimal_k <- K_range[which.max(result_Sil)]
  # result[[which.max(result_Sil)]]@cluster
  
  output_label[seq_iter,] <- result@cluster
  
  
}


dir.create( paste0("result/c3_",scenario,"_n",num_node), showWarnings = FALSE)

write.csv(output_label, 
          paste0("result/c3_",scenario,"_n",num_node,"/output_label.csv"),
          row.names = F)


