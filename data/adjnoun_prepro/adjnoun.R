library(networkdata)
library(igraph)
library(reticulate)
np <- import("numpy")
data(adjnoun)

adj <- as.matrix(as_adjacency_matrix(adjnoun))
node_labels <- V(adjnoun)$name
node_classes <- V(adjnoun)$value 

adj <- np$array(adj)
node_labels <- np$array(node_labels)
node_classes <- np$array(node_classes)

np$savez("adjnoun_raw.npz", adj=adj, labels=node_labels, classes=node_classes)
