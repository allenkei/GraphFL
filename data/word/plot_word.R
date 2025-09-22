

library(ggplot2)
library(dplyr)


word_cluster <- readLines(file.choose()) # GFL cluster result
word_mu <- read.csv(file.choose(), header=F) # GFL estimated mu

node_labels <- rep(0, 112)
for (i in seq_along(word_cluster)) {
  nodes <- as.numeric(strsplit(word_cluster[i], "\\s+")[[1]]) + 1
  node_labels[nodes] <- i
}

colnames(word_mu) <- c("dim1", "dim2", "dim3")
word_mu$label <- as.factor(node_labels) 




#################################
# 3D latent space visualization #
#################################

cluster_colors <- c(
  "1" = "deepskyblue",
  "2" = "orange",
  "3" = "red",
  "4" = "darkred")



cluster_levels <- sort(unique(node_labels))
cluster_colors <- setNames(cluster_colors[1:length(cluster_levels)], cluster_levels)





par(mfrow=c(1,3),mar=c(1/2,1/2,1/2,1/2))
scatter3D(
  x = word_mu$dim1, y = word_mu$dim2, z = word_mu$dim3,
  colvar = as.numeric(word_mu$label), col = cluster_colors,
  pch = 19, cex = 1, alpha = 0.8,
  xlab = "dim1", ylab = "dim2", zlab = "dim3",
  theta = 20, phi = 30,   # viewing angle
  bty = "g", ticktype = "detailed", colkey = FALSE
)
scatter3D(
  x = word_mu$dim1, y = word_mu$dim2, z = word_mu$dim3,
  colvar = as.numeric(word_mu$label), col = cluster_colors,
  pch = 19, cex = 1, alpha = 0.8,
  xlab = "dim1", ylab = "dim2", zlab = "dim3",
  theta = 60, phi = 40,   # viewing angle
  bty = "g", ticktype = "detailed", colkey = FALSE
)
scatter3D(
  x = word_mu$dim1, y = word_mu$dim2, z = word_mu$dim3,
  colvar = as.numeric(word_mu$label), col = cluster_colors,
  pch = 19, cex = 1, alpha = 0.8,
  xlab = "dim1", ylab = "dim2", zlab = "dim3",
  theta = 50, phi = 60,   # viewing angle
  bty = "g", ticktype = "detailed", colkey = FALSE
)



base_theme <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_blank(),
    legend.position = "none"
  )


p1_xy <- ggplot(word_mu, aes(x = dim1, y = dim2, color = label)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = cluster_colors) +
  base_theme + labs(x = "dim1", y = "dim2")


p1_xz <- ggplot(word_mu, aes(x = dim1, y = dim3, color = label)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = cluster_colors) +
  base_theme + labs(x = "dim1", y = "dim3")


p1_yz <- ggplot(word_mu, aes(x = dim2, y = dim3, color = label)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = cluster_colors) +
  base_theme + labs(x = "dim2", y = "dim3")


(p1_xy | p1_xz | p1_yz)

##############
# word table #
##############

nodes <- read_csv("data/word/nodes.csv")
word_cluster <- readLines(file.choose()) # GFL cluster result



cluster_indices <- lapply(word_cluster, function(x) as.integer(strsplit(x, " ")[[1]]))


word_clusters <- lapply(cluster_indices, function(indices) {
  filter(nodes, `# index` %in% indices) %>% pull(label)
})

# Name the list by cluster
names(word_clusters) <- paste0("Cluster_", seq_along(word_clusters))

# Example usage
word_clusters$Cluster_1  
word_clusters$Cluster_2  
word_clusters$Cluster_3  
word_clusters$Cluster_4  






