library(reticulate)
library(dtwclust)

# load data
np <- import("numpy")
#TS_data <- np$load("data/data_s1_n120.npz")
TS_data <- np$load("data/data_s2_n144.npz")

num_node <- "144" # "120", "144"
scenario <- "s2"  # "s1", "s2"


data <- TS_data$f["y"]  # seq by nodes by T
label <- TS_data$f["labels"]

output_label <- array(NA, dim = dim(label)) # saved output labels

T_seq <- dim(data)[1]
K_range <- 2:10

for(seq_iter in 1:T_seq){
  
  one_data <- data[seq_iter,,]
  
  result <- tsclust(one_data, 
                    k = K_range,
                    distance = "L2",
                    type = "partitional",
                    seed = 123)
  
  result_Sil <- sapply(result, cvi, type = "Sil")
  #optimal_k <- K_range[which.max(result_Sil)]
  
  output_label[seq_iter,] <- result[[which.max(result_Sil)]]@cluster
  
  
}

output_label

write.csv(output_label, 
          paste0("result/c3_",scenario,"_n",num_node,"/output_label.csv"),
          row.names = F)


