library(reticulate)
library(ggplot2)
library(igraph)
library(patchwork)
np <- reticulate::import("numpy")


adj_mat1 <- "data/data_s1_n120.npz"
adj_mat2 <- "data/data_s2_n144.npz" 


data1 <- np$load(adj_mat1)
data2 <- np$load(adj_mat2)
adj_mat1 <- data1['adj_matrices'][1,,]
adj_mat2 <- data2['adj_matrices'][1,,]
label1 <- data1['labels'][1,]
label2 <- data2['labels'][1,]
rm(data1,data2)

#dim(adj_mat1); dim(adj_mat2)




##################
# Simulated data #
##################





graph <- graph_from_adjacency_matrix(adj_mat1, mode = "undirected", diag = FALSE)

community_colors <- c("skyblue", "orange", "green")  
node_colors <- community_colors[as.factor(label1)]


comm_1_nodes <- which(label1 == 0)  
comm_2_nodes <- which(label1 == 1)  
comm_3_nodes <- which(label1 == 2)  
layout <- matrix(NA, nrow = length(label1), ncol = 2)
layout[comm_1_nodes, ] <- cbind(runif(length(comm_1_nodes), min = -0.2, max = 0.2), # range for x
                                runif(length(comm_1_nodes), min = 0.7, max = 1.2))  # range for y

layout[comm_2_nodes, ] <- cbind(runif(length(comm_2_nodes), min = -0.6, max = -0.2), # range for x
                                runif(length(comm_2_nodes), min = -1.3, max = -0.5)) # range for y

layout[comm_3_nodes, ] <- cbind(runif(length(comm_3_nodes), min = 0.2, max = 0.6),   # range for x
                                runif(length(comm_3_nodes), min = -1.3, max = -0.5)) # range for y





graph2 <- graph_from_adjacency_matrix(adj_mat2, mode = "undirected", diag = FALSE)
cluster_colors2 <- colorRampPalette(c("deepskyblue","lightgreen","coral", "orange"))(4)
node_colors2 <- cluster_colors2[as.factor(label2)]




# 10 by 5
par(mfrow=c(1,2), mar = c(1,1,1,1))
set.seed(123) # needed for node location in figures
plot(graph, 
     vertex.size = 5, vertex.label = NA, vertex.color = node_colors,  
     layout = layout, edge.color = "lightgray",  
     main = "Block Graph with 3 Clusters",
     xlim = c(-1, 1),        
     ylim = c(-1, 1)) 


plot(graph2, vertex.size = 5, vertex.label = NA, 
     vertex.color = node_colors2,
     main = "Grid Graph with 4 Clusters")







#par(mfrow=c(1,2), mar = c(2,2,2,2)) # adjacency matrix
#image(adj_mat1, xaxt = "n", yaxt = "n", main="Graph with 3 Clusters")
#axis(side=1,at=seq(0,1,length.out = 4),labels=c(1,40,80,120),xpd=NA,cex.axis=1)
#axis(side=2,at=seq(0,1,length.out = 4),labels=c(1,40,80,120),xpd=NA,cex.axis=1)
#image(adj_mat2, xaxt = "n", yaxt = "n", main = "Grid Graph with 4 Clusters")
#axis(side=1,at=seq(0,1,length.out = 4),labels=c(1,48,96,144),xpd=NA,cex.axis=1)
#axis(side=2,at=seq(0,1,length.out = 4),labels=c(1,48,96,144),xpd=NA,cex.axis=1)


################
# Estimated mu #
################

mu_data1 <- read.csv(file.choose(), header = FALSE) # choose the file (s1)
mu_data2 <- read.csv(file.choose(), header = FALSE) # choose the file (s2)
colnames(mu_data1) <- colnames(mu_data2) <- c("dim1", "dim2", "dim3")


mu_data1$label1 <- as.factor(label1); mu_data2$label2 <- as.factor(label2)


###########################
# Projection with 3 Views #
###########################
base_theme <- theme_minimal(base_size = 12) + theme(panel.grid.minor = element_blank(),
        plot.title = element_blank(), legend.position = "none")

# mu from s1
p1_xy <- ggplot(mu_data1, aes(x = dim1, y = dim2, color = label1)) +
  geom_point(alpha = 0.8, size = 1.8) +
  base_theme + labs(x = "dim1", y = "dim2")

p1_xz <- ggplot(mu_data1, aes(x = dim1, y = dim3, color = label1)) +
  geom_point(alpha = 0.8, size = 1.8) +
  base_theme + labs(x = "dim1", y = "dim3")

p1_yz <- ggplot(mu_data1, aes(x = dim2, y = dim3, color = label1)) +
  geom_point(alpha = 0.8, size = 1.8) +
  base_theme + labs(x = "dim2", y = "dim3")

# mu from s2
p2_xy <- ggplot(mu_data2, aes(x = dim1, y = dim2, color = label2)) +
  geom_point(alpha = 0.8, size = 1.8) +
  base_theme + labs(x = "dim1", y = "dim2")

p2_xz <- ggplot(mu_data2, aes(x = dim1, y = dim3, color = label2)) +
  geom_point(alpha = 0.8, size = 1.8) +
  base_theme + labs(x = "dim1", y = "dim3")

p2_yz <- ggplot(mu_data2, aes(x = dim2, y = dim3, color = label2)) +
  geom_point(alpha = 0.8, size = 1.8) +
  base_theme + labs(x = "dim2", y = "dim3")

(p1_xy | p1_xz | p1_yz) / (p2_xy | p2_xz | p2_yz)



###################
# 3D Scatter Plot #
###################

library(plot3D)

par(mfrow=c(1,2), mar = c(1,1,1,1)) 
scatter3D(
  x = mu_data1$dim1, y = mu_data1$dim2, z = mu_data1$dim3,
  colvar = as.numeric(mu_data1$label1),    # color by label
  col = rainbow(length(unique(mu_data1$label1))),  # color palette
  pch = 19, cex = 1, alpha = 0.8,
  xlab = "dim1", ylab = "dim2", zlab = "dim3",
  theta = 20, phi = 20,   # viewing angle
  bty = "g",
  ticktype = "detailed", 
  colkey = F
)


scatter3D(
  x = mu_data2$dim1, y = mu_data2$dim2, z = mu_data2$dim3,
  colvar = as.numeric(mu_data2$label2),    # color by label
  col = rainbow(length(unique(mu_data2$label2))),  # color palette
  pch = 19, cex = 1, alpha = 0.8,
  xlab = "dim1", ylab = "dim2", zlab = "dim3",
  theta = 20, phi = 20,   # viewing angle
  bty = "g",
  ticktype = "detailed",     
  colkey = FALSE
)


# Simple Version
#scatterplot3d(mu_data1$dim1, mu_data1$dim2, mu_data1$dim3, 
#              xlab = "dim1", ylab = "dim2", zlab = "dim3", pch = 19, color = "blue")
