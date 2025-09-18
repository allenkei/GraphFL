library(reticulate)
library(ggplot2)
library(igraph)
library(patchwork)
np <- reticulate::import("numpy")


adj_mat1 <- "data/data_s1_n120.npz"
adj_mat2 <- "data/data_s2_n144.npz"
adj_mat3 <- "data/data_s3_n120.npz" 


data1 <- np$load(adj_mat1)
data2 <- np$load(adj_mat2)
data3 <- np$load(adj_mat3)
adj_mat1 <- data1['adj_matrices'][1,,]
adj_mat2 <- data2['adj_matrices'][1,,]
adj_mat3 <- data3['adj_matrices'][1,,]
label1 <- data1['labels'][1,]
label2 <- data2['labels'][1,]
label3 <- data3['labels'][1,]
ts1 <- data1['y'][1,,]
ts2 <- data2['y'][1,,]
ts3 <- data3['y'][1,,]
rm(data1,data2,data3) #dim(adj_mat1); dim(adj_mat2); dim(adj_mat3)

community_colors1 <- c("deepskyblue", "green", "orange")  
community_colors2 <- c("deepskyblue", "green", "red", "orange")
community_colors3 <- c("deepskyblue", "green", "orange")  

#########
# Fig 1 #
#########
# 10 by 3.5

graph <- graph_from_adjacency_matrix(adj_mat1, mode = "undirected", diag = FALSE)
node_colors1 <- community_colors1[as.factor(label1)]
comm_1_nodes <- which(label1==0);comm_2_nodes <- which(label1==1);comm_3_nodes <- which(label1==2)  
layout <- matrix(NA, nrow = length(label1), ncol = 2)
layout[comm_1_nodes,] <- cbind(runif(length(comm_1_nodes), min = -0.2, max = 0.2), # range for x
                               runif(length(comm_1_nodes), min = 0.7, max = 1.2))  # range for y
layout[comm_2_nodes,] <- cbind(runif(length(comm_2_nodes), min = -0.6, max = -0.2), # range for x
                               runif(length(comm_2_nodes), min = -1.3, max = -0.5)) # range for y
layout[comm_3_nodes,] <- cbind(runif(length(comm_3_nodes), min = 0.2, max = 0.6),   # range for x
                               runif(length(comm_3_nodes), min = -1.3, max = -0.5)) # range for y



layout(matrix(c(1, 2), nrow = 1, ncol = 2, byrow = TRUE), widths = c(1, 2), heights = c(1, 1))

par(mar = c(1, 1, 1, 1)); set.seed(123)
plot(graph, 
     vertex.size = 5, vertex.label = NA, vertex.color = node_colors1, vertex.frame.color = NA,
     layout = layout, edge.color = "lightgray", edge.width = 0.2,
     main = "Block Graph with 3 Clusters",
     xlim = c(-1, 1), ylim = c(-1, 1)) 

par(mar = c(4, 4, 2, 1), cex.axis = 0.8)
clusters1 <- sort(unique(label1)); Tn <- ncol(ts1); use_sem <- FALSE

plot(NA, xlim = c(1, Tn), ylim = range(-3,3),# axes = FALSE,
     xlab = "Time point", ylab = "Value", main = "Time Series Means ± SD")
#axis(1); axis(2, at = c(-3, -1, 1, 3))
for (i in seq_along(clusters1)) {
  k <- clusters1[i]
  X <- ts1[label1 == k, , drop = FALSE]
  m <- colMeans(X); s <- apply(X, 2, sd)
  b <- if (use_sem) s / sqrt(nrow(X)) else s
  polygon(c(1:Tn, Tn:1), c(m + b, rev(m - b)),
          col = adjustcolor(community_colors1[i], alpha.f = 0.1), border = NA)
  lines(m, col = community_colors1[i], lwd = 2)
}


#########
# Fig 2 #
#########
# 10 by 3.5

graph2 <- graph_from_adjacency_matrix(adj_mat2, mode = "undirected", diag = FALSE)
cluster_colors2 <- colorRampPalette(c("deepskyblue","green","red", "orange"))(4)
node_colors2 <- cluster_colors2[as.factor(label2)]

layout(matrix(c(1, 2), nrow = 1, ncol = 2, byrow = TRUE), widths = c(1, 2), heights = c(1, 1))


par(mar = c(1, 1, 1, 1)); set.seed(123)
plot(graph2, vertex.size = 5, vertex.label = NA, vertex.frame.color = NA,
     vertex.color = node_colors2, main = "Grid Graph with 4 Clusters")

par(mar = c(4, 4, 2, 1), cex.axis = 0.8)
clusters2 <- sort(unique(label2)); Tn <- ncol(ts2); use_sem <- FALSE

plot(NA, xlim = c(1, Tn), ylim = range(-3,3.5), #axes = FALSE,
     xlab = "Time point", ylab = "Value", main = "Time Series Means ± SD")
#axis(1); axis(2, at = c(-3, -1, 1, 3))
for (i in seq_along(clusters2)) {
  k <- clusters2[i]
  X <- ts2[label2 == k, , drop = FALSE]
  m <- colMeans(X); s <- apply(X, 2, sd)
  b <- if (use_sem) s / sqrt(nrow(X)) else s
  
  polygon(c(1:Tn, Tn:1), c(m + b, rev(m - b)),
          col = adjustcolor(community_colors2[i], alpha.f = 0.1), border = NA)
  lines(m, col = community_colors2[i], lwd = 2)
}


#########
# Fig 3 #
#########
# 10 by 3.5

graph3 <- graph_from_adjacency_matrix(adj_mat3, mode = "undirected", diag = FALSE)
node_colors3 <- community_colors3[as.factor(label3)]
comm_1_nodes <- which(label3==0);comm_2_nodes <- which(label3==1);comm_3_nodes <- which(label3==2)  
layout <- matrix(NA, nrow = length(label3), ncol = 2)
layout[comm_1_nodes,] <- cbind(runif(length(comm_1_nodes), min = -0.2, max = 0.2), # range for x
                               runif(length(comm_1_nodes), min = 0.7, max = 1.2))  # range for y
layout[comm_2_nodes,] <- cbind(runif(length(comm_2_nodes), min = -0.6, max = -0.2), # range for x
                               runif(length(comm_2_nodes), min = -1.3, max = -0.5)) # range for y
layout[comm_3_nodes,] <- cbind(runif(length(comm_3_nodes), min = 0.2, max = 0.6),   # range for x
                               runif(length(comm_3_nodes), min = -1.3, max = -0.5)) # range for y



layout(matrix(c(1, 2), nrow = 1, ncol = 2, byrow = TRUE), widths = c(1, 2), heights = c(1, 1))

par(mar = c(1, 1, 1, 1)); set.seed(123)
plot(graph3, 
     vertex.size = 5, vertex.label = NA, vertex.color = node_colors3, vertex.frame.color = NA,
     layout = layout, edge.color = "lightgray", edge.width = 0.2,
     main = "Block Graph with 3 Clusters",
     xlim = c(-1, 1), ylim = c(-1, 1)) 

par(mar = c(4, 4, 2, 1), cex.axis = 0.8)
clusters3 <- sort(unique(label3)); Tn <- ncol(ts3); use_sem <- FALSE

plot(NA, xlim = c(1, Tn), ylim = range(-4,4),# axes = FALSE,
     xlab = "Time point", ylab = "Value", main = "Time Series Means ± SD")
#axis(1); axis(2, at = c(-3, -1, 1, 3))
for (i in seq_along(clusters3)) {
  k <- clusters3[i]
  X <- ts3[label3 == k, , drop = FALSE]
  m <- colMeans(X); s <- apply(X, 2, sd)
  b <- if (use_sem) s / sqrt(nrow(X)) else s
  polygon(c(1:Tn, Tn:1), c(m + b, rev(m - b)),
          col = adjustcolor(community_colors3[i], alpha.f = 0.1), border = NA)
  lines(m, col = community_colors3[i], lwd = 2)
}

################
# Estimated mu #
################


adj_mat1 <- "data/data_s1_n210.npz"
adj_mat2 <- "data/data_s2_n196.npz" 
adj_mat3 <- "data/data_s3_n210.npz" 

np <- reticulate::import("numpy")
data1 <- np$load(adj_mat1)
data2 <- np$load(adj_mat2)
data3 <- np$load(adj_mat3)
label1 <- data1['labels'][1,]
label2 <- data2['labels'][1,]
label3 <- data3['labels'][1,]
rm(data1,data2,data3)
community_colors1 <- c("deepskyblue", "green", "orange")  
community_colors2 <- c("deepskyblue", "green", "coral", "orange")
community_colors3 <- c("deepskyblue", "green", "orange")

# LOAD LEARNED MU FROM result FOLDER
mu_data1 <- read.csv(file.choose(), header = FALSE) # choose the file (s1)
mu_data2 <- read.csv(file.choose(), header = FALSE) # choose the file (s2)
mu_data3 <- read.csv(file.choose(), header = FALSE) # choose the file (s3)

mu_data1$label_col1 <- as.factor(community_colors1[label1+1])
mu_data2$label_col2 <- as.factor(community_colors2[label2+1])
mu_data3$label_col3 <- as.factor(community_colors3[label3+1])

colnames(mu_data1) <- colnames(mu_data2) <- colnames(mu_data3) <- c("dim1", "dim2", "dim3", "label_col")



###########################
# Projection with 3 Views #
###########################
base_theme <- theme_minimal(base_size = 12) + theme(panel.grid.minor = element_blank(),
                                                    plot.title = element_blank(), legend.position = "none")

# mu from s1
p1_xy <- ggplot(mu_data1, aes(x = dim1, y = dim2, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim1", y = "dim2")

p1_xz <- ggplot(mu_data1, aes(x = dim1, y = dim3, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim1", y = "dim3")

p1_yz <- ggplot(mu_data1, aes(x = dim2, y = dim3, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim2", y = "dim3")

# mu from s2
p2_xy <- ggplot(mu_data2, aes(x = dim1, y = dim2, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim1", y = "dim2")

p2_xz <- ggplot(mu_data2, aes(x = dim1, y = dim3, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim1", y = "dim3")

p2_yz <- ggplot(mu_data2, aes(x = dim2, y = dim3, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim2", y = "dim3")

# mu from s3
p3_xy <- ggplot(mu_data3, aes(x = dim1, y = dim2, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim1", y = "dim2")

p3_xz <- ggplot(mu_data3, aes(x = dim1, y = dim3, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim1", y = "dim3")

p3_yz <- ggplot(mu_data3, aes(x = dim2, y = dim3, color = label_col)) +
  geom_point(alpha = 0.8, size = 1.8) + scale_color_identity() +
  base_theme + labs(x = "dim2", y = "dim3")

# 9 by 6
(p1_xy | p1_xz | p1_yz) / (p2_xy | p2_xz | p2_yz) / (p3_xy | p3_xz | p3_yz)



###################
# 3D Scatter Plot #
###################

library(plot3D)

par(mfrow=c(1,3), mar = c(1,1,1,1)) 
scatter3D(
  x = mu_data1$dim1, y = mu_data1$dim2, z = mu_data1$dim3,
  colvar = as.numeric(mu_data1$label_col),
  col = community_colors1,
  pch = 19, cex = 1, alpha = 0.8,
  xlab = "dim1", ylab = "dim2", zlab = "dim3",
  theta = 20 , phi = 20,   # viewing angle
  bty = "g", ticktype = "detailed", colkey = F
)


scatter3D(
  x = mu_data2$dim1, y = mu_data2$dim2, z = mu_data2$dim3,
  colvar = as.numeric(mu_data2$label_col),
  col = community_colors2,
  pch = 19, cex = 1, alpha = 0.8,
  xlab = "dim1", ylab = "dim2", zlab = "dim3",
  theta = 20, phi = 20,   # viewing angle
  bty = "g", ticktype = "detailed", colkey = FALSE
)



scatter3D(
  x = mu_data3$dim1, y = mu_data3$dim2, z = mu_data3$dim3,
  colvar = as.numeric(mu_data3$label_col),
  col = community_colors3,
  pch = 19, cex = 1, alpha = 0.8,
  xlab = "dim1", ylab = "dim2", zlab = "dim3",
  theta = 20, phi = 20,   # viewing angle
  bty = "g", ticktype = "detailed", colkey = FALSE
)

# Simple Version
#scatterplot3d(mu_data1$dim1, mu_data1$dim2, mu_data1$dim3, 
#              xlab = "dim1", ylab = "dim2", zlab = "dim3", pch = 19, color = "blue")
