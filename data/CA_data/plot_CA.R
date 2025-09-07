

library(sf)
library(ggplot2)
library(dplyr)
shapefile_path <- "data/CA_data/California_Counties.shp"
ca_map <- st_read(shapefile_path) # ca_map$NAME gives county names
ca_map$NAME <- gsub("(?i) County", "", ca_map$NAME, perl = TRUE)

CA_cluster <- readLines(file.choose()) # GFL cluster result
CA_mu <- read.csv(file.choose(), header=F) # GFL estimated mu

node_labels <- rep(0, 58)
for (i in seq_along(CA_cluster)) {
  nodes <- as.numeric(strsplit(CA_cluster[i], "\\s+")[[1]]) + 1
  node_labels[nodes] <- i
}

colnames(CA_mu) <- c("dim1", "dim2", "dim3")
CA_mu$label <- as.factor(node_labels) 
ca_map$cluster <- node_labels # add label to map


###################
# CA map from GFL #
###################
tableau20 <- c(
  "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
  "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
  "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
  "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5")

cluster_levels <- sort(unique(ca_map$cluster))
cluster_colors <- setNames(tableau20[1:length(cluster_levels)], cluster_levels)


p_map <- ggplot(ca_map) +
  geom_sf(aes(fill = as.factor(cluster)), color = "white", size = 0.2) +
  geom_sf_text(aes(label = NAME), size = 2, color = "black") +
  scale_fill_manual(name = "Cluster", values = cluster_colors) +
  theme_minimal() +
  theme(
    axis.title = element_blank(),      
    axis.text = element_blank(),       
    axis.ticks = element_blank(),      
    panel.grid = element_blank()       
  );p_map


#################################
# 3D latent space visualization #
#################################

base_theme <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_blank(),
    legend.position = "none"
  )


p1_xy <- ggplot(CA_mu, aes(x = dim1, y = dim2, color = label)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = cluster_colors) +
  base_theme + labs(x = "dim1", y = "dim2")


p1_xz <- ggplot(CA_mu, aes(x = dim1, y = dim3, color = label)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = cluster_colors) +
  base_theme + labs(x = "dim1", y = "dim3")


p1_yz <- ggplot(CA_mu, aes(x = dim2, y = dim3, color = label)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = cluster_colors) +
  base_theme + labs(x = "dim2", y = "dim3")


(p1_xy | p1_xz | p1_yz) 

######################
# CA map from kmeans #
######################

CA_data <- read.csv("data/CA_data/CA_time_series.csv")


set.seed(42) # different seeds, different results
kmean_result <- kmeans(CA_data, centers = 10)
kmean_result$cluster


shapefile_path <- "data/CA_data/California_Counties.shp"
ca_map <- st_read(shapefile_path) # ca_map$NAME gives county names
ca_map$NAME <- gsub("(?i) County", "", ca_map$NAME, perl = TRUE)

ca_map$cluster <- kmean_result$cluster # add KMEAN label to map

tableau20 <- c(
  "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
  "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
  "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
  "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5")

cluster_levels <- sort(unique(ca_map$cluster))
cluster_colors <- setNames(tableau20[1:length(cluster_levels)], cluster_levels)


p_map <- ggplot(ca_map) +
  geom_sf(aes(fill = as.factor(cluster)), color = "white", size = 0.2) +
  geom_sf_text(aes(label = NAME), size = 2, color = "black") +
  scale_fill_manual(name = "Cluster", values = cluster_colors) +
  theme_minimal() +
  theme(
    axis.title = element_blank(),      
    axis.text = element_blank(),       
    axis.ticks = element_blank(),      
    panel.grid = element_blank()       
  );p_map


