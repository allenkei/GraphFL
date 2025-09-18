library(readr)
nodes <- read_csv("~/Downloads/network/nodes.csv")
edges <- read_csv("~/Downloads/network/edges.csv")

n_chunks <- 100 # length of time
g <- graph_from_data_frame(d = edges, vertices = nodes, directed = FALSE)
A <- as_adjacency_matrix(g, sparse = FALSE) # 112 by 112
colnames(A) <- rownames(A) <- nodes$label




library(tidyverse)

file_name <- "~/Downloads/network/david_copperfield.txt" # david_copperfield.txt

book_text <- tolower(readChar(file_name, file.info(file_name)$size))

# Split into words
tokens <- unlist(strsplit(book_text, "\\W+"))   # split on non-word characters
tokens <- tokens[tokens != ""]                  # remove empty

# Divide by chunks
chunk_size <- ceiling(length(tokens) / n_chunks)
chunks <- split(tokens, ceiling(seq_along(tokens)/chunk_size))

# Count words in each chunks
word_series <- matrix(0, nrow = length(nodes$label), ncol = n_chunks,
                      dimnames = list(nodes$label, paste0("t", 1:n_chunks)))

for (j in seq_along(chunks)) {
  tab <- table(chunks[[j]])
  common <- intersect(names(tab), nodes$label)
  word_series[common, j] <- as.numeric(tab[common])
}

word_zscore <- t(scale(t(word_series)))  # row-wise scaling

#write.csv(word_zscore, "data/word/word_time_series.csv", row.names = TRUE)
#write.csv(A, "data/word/word_graph.csv", row.names = TRUE)

#################
# Visualization #
#################

nodes <- read_csv("~/Downloads/network/nodes.csv")
edges <- read_csv("~/Downloads/network/edges.csv")
word_cluster <- readLines(file.choose()) # GFL cluster result

node_labels <- rep(0, 112) # there are 112 words
for (i in seq_along(word_cluster)) {
  nodes_idx <- as.numeric(strsplit(word_cluster[i], "\\s+")[[1]]) + 1
  node_labels[nodes_idx] <- i
}

g <- graph_from_data_frame(d = edges, vertices = nodes, directed = FALSE)
V(g)$name <- as.factor(node_labels)


cluster_colors <- c(
  "1" = "deepskyblue",
  "2" = "orange",
  "3" = "red",
  "4" = "firebrick",
  "5" = "darkred")

v_cols <- cluster_colors[as.character(V(g)$name)]
v_shapes <- ifelse(nodes$value == 1, "circle", "square")




par(mar = c(0.5, 0.5, 0.5, 0.5))  

set.seed(42)
lo <- layout_with_fr(g) # "fr", "kk", "lgl"

plot(
  g, layout = lo,
  vertex.color = v_cols,
  vertex.shape = v_shapes,
  vertex.size = 6,
  vertex.label = NA,
  edge.color = "grey70",
  edge.width = 1,
  rescale = TRUE,         # use full plot area
  #asp = 0.5,                # allow stretching in both x/y
  #main = "Noun & Adjective Network"
)




# Place shape legend top right
legend(
  "topright",
  legend = c("Noun", "Adjective"),
  pch = c(16, 15),    # circle and square
  col = "grey30",
  pt.cex = 1.5,
  bty = "n",
  title = "Shape & Color",
  inset = c(0.0, 0.1)    # lower inset → below the first legend
)

legend(
  "topright",
  legend = paste("Cluster", 1:5),
  text.col = cluster_colors,   # text colored by cluster
  bty = "n", 
  inset = c(0.03, 0.2)
)
