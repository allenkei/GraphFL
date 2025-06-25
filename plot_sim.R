library(reticulate)
library(ggplot2)
library(igraph)
library(pheatmap)
#library(RColorBrewer)
#library(viridis)
np <- reticulate::import("numpy")


adj_mat1 <- "data/data_s1_n120.npz"
adj_mat2 <- "data/data_s2_n196.npz" 


data1 <- np$load(adj_mat1)
data2 <- np$load(adj_mat2)
adj_mat1 <- data1['adj_matrices'][1,,]
adj_mat2 <- data2['adj_matrices'][1,,]
label1 <- data1['labels'][1,]
label2 <- data2['labels'][1,]
rm(data1,data2)

#dim(adj_mat1); dim(adj_mat2)







par(mfrow=c(1,2))
par(mar = c(1,1,1,1))



graph <- graph_from_adjacency_matrix(adj_mat1, mode = "undirected", diag = FALSE)

community_colors <- c("skyblue", "orange", "green")  
node_colors <- community_colors[as.factor(label1)]


comm_1_nodes <- which(label1 == 0)  
comm_2_nodes <- which(label1 == 1)  
comm_3_nodes <- which(label1 == 2)  
layout <- matrix(NA, nrow = length(label1), ncol = 2)
layout[comm_1_nodes, ] <- cbind(runif(length(comm_1_nodes), min = -0.2, max = 0.2),  # Tighter range for x
                                runif(length(comm_1_nodes), min = 0.7, max = 1.2))  # Higher y values for the top

layout[comm_2_nodes, ] <- cbind(runif(length(comm_2_nodes), min = -0.6, max = -0.2),  # Wider range for x
                                runif(length(comm_2_nodes), min = -1.3, max = -0.5))  # Lower y values for bottom-left

layout[comm_3_nodes, ] <- cbind(runif(length(comm_3_nodes), min = 0.2, max = 0.6),  # Wider range for x
                                runif(length(comm_3_nodes), min = -1.3, max = -0.5))  # Lower y values for bottom-right





graph2 <- graph_from_adjacency_matrix(adj_mat2, mode = "undirected", diag = FALSE)
cluster_colors2 <- colorRampPalette(c("deepskyblue","lightgreen","coral", "orange"))(4)
node_colors2 <- cluster_colors2[as.factor(label2)]




# 10 by 5
set.seed(3) # needed for node location in figures
plot(graph, 
     vertex.size = 5, vertex.label = NA, vertex.color = node_colors,  
     layout = layout, edge.color = "lightgray",  
     main = "Graph with 3 Clusters",
     xlim = c(-1, 1),        
     ylim = c(-1, 1)) 


plot(graph2, vertex.size = 5, vertex.label = NA, 
     vertex.color = node_colors2,
     main = "Grid Graph with 4 Clusters")







par(mfrow=c(1,2))
par(mar = c(2,2,2,2))

image(adj_mat1, xaxt = "n", yaxt = "n", main="Graph with 3 Clusters")
axis(side=1,at=seq(0,1,length.out = 4),labels=c(1,40,80,120),xpd=NA,cex.axis=1)
axis(side=2,at=seq(0,1,length.out = 4),labels=c(1,40,80,120),xpd=NA,cex.axis=1)

image(adj_mat2, xaxt = "n", yaxt = "n", main = "Grid Graph with 4 Clusters")
axis(side=1,at=seq(0,1,length.out = 4),labels=c(1,65,130,196),xpd=NA,cex.axis=1)
axis(side=2,at=seq(0,1,length.out = 4),labels=c(1,65,130,196),xpd=NA,cex.axis=1)




#pheatmap(adj_mat1, cluster_rows = FALSE, cluster_cols = FALSE, 
#         main = "Heatmap of Adjacency Matrix", 
#         color = colorRampPalette(c("white", "blue"))(100))


