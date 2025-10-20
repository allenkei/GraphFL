rm(list = ls())

# in windows code
# setwd("D:/onedrive/ucsc phd/Research/network theory/climate time series network")

# in mac code
setwd("/Users/xichen/Library/CloudStorage/OneDrive-Personal/ucsc phd/Research/network theory/climate time series network")



library(vars)
library(MASS)
library(reticulate)
library(ggplot2)
library(reshape2)
library(NAC)
library(igraph)
library(clue)
library(mclust)

np <- import("numpy")

simulate_var1_randomPhi_blockSigma <- function(
    T = 100,
    mus = c(-1, 0, 1),                # mean per cluster
    cluster_sizes = c(30, 40, 50),   # 1..30, 31..70, 71..120
    phi_sd = 0.10,                   # spread of random Phi entries before shrinking
    target_rho = 0.98,               # cap spectral radius of Phi
    sigma2 = c(1, 1, 1),             # innovation variance by cluster
    rho_eps = c(0.5, 0.5, 0.5),   # within-cluster corr in Sigma blocks
    burn_in = 500,
    random_phi = FALSE
){
  stopifnot(length(cluster_sizes) == 3, length(mus) == 3)
  stopifnot(length(sigma2) == 3, length(rho_eps) == 3)
  
  K <- 3
  n <- sum(cluster_sizes)
  idx_start <- cumsum(c(1, head(cluster_sizes, -1)))
  idx_end   <- cumsum(cluster_sizes)
  
  # --- labels and means ---
  z  <- rep(1:(K), times = cluster_sizes)
  mu <- mus[z]
  
  if (random_phi) {
    # random Phi, then enforce stationarity by spectral shrinkage ---
    Phi_raw <- matrix(rnorm(n*n, mean = 0, sd = phi_sd), n, n)
    ev <- eigen(Phi_raw, only.values = TRUE)$values
    rho <- max(Mod(ev))
    if (rho >= target_rho) {
      Phi <- Phi_raw * (target_rho / rho)
    } else {
      Phi <- Phi_raw
    }
  } else {
  # alternative: diagonal Phi
    # Phi <- diag(1 / (1 + 1:n))
    Phi <- diag(rep(0.5, n))
  }
  # --- block-diagonal Sigma with compound symmetry in each cluster ---
  make_sigma_block <- function(m, sig2, r) {
    # PSD if r in (-1/(m-1), 1)
    stopifnot(r > -1/(m-1) && r < 1)
    (sig2 * (1 - r)) * diag(m) + (sig2 * r) * matrix(1, m, m)
  }
  Sigma <- matrix(0, n, n)
  for (k in 1:K) {
    i1 <- idx_start[k]; i2 <- idx_end[k]; m <- cluster_sizes[k]
    Sigma[i1:i2, i1:i2] <- make_sigma_block(m, sigma2[k], rho_eps[k])
  }
  
  # --- first observation calibration
  # vec_Sy <- solve(diag(n*n) - kronecker(Phi, Phi)) %*% as.vector(Sigma)
  # Sigma_y <- matrix(vec_Sy, n, n)
  
  # --- simulate VAR(1) ---
  Y <- matrix(0, ncol = burn_in + T, nrow = n)
  y <- rep(0, n)
  for (t in 1:(burn_in + T)) {
    eps <- mvrnorm(1, mu = rep(0, n), Sigma = Sigma)
    y <- Phi %*% (y-mu) + mu + eps
    Y[,t] <- y
  }
  X <- Y[,(burn_in + 1):(burn_in + T)]
  
  list(
    y = X,                     # 120 x T matrix
    Phi = Phi,                 # 120 x 120 random, stationary
    Sigma = Sigma,             # 120 x 120 block-diagonal
    labels = as.integer(z-1),                     # true cluster labels (0..2)
    mu = mu
  )
}

sbm_adjacency <- function(cluster_sizes = c(30,40,50),
                          p_in = 0.30, p_out = 0.15) {
  K <- length(cluster_sizes)
  z <- rep(seq_len(K), times = cluster_sizes)
  n <- sum(cluster_sizes)
  
  A <- matrix(NA, n, n)  # initialize with zeros
  
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      p <- if (z[i] == z[j]) p_in else p_out
      edge <- rbinom(1, 1, p)
      A[i, j] <- edge
      A[j, i] <- edge   # make symmetric
    }
  }
  
  diag(A) <- 0L   # no self-loops
  list(adj_matrices = A)
}


plot_var_data <- function(X, z, nodes = 1:5, title = "Simulated VAR(1) Data") {
  # X is T × n (time × nodes)
  df <- data.frame(Time = 1:nrow(X), X[, nodes, drop = FALSE])
  df_long <- melt(df, id.vars = "Time", variable.name = "Node", value.name = "Value")
  df_long$Cluster <- factor(z[nodes])
  
  ggplot(df_long, aes(x = Time, y = Value, color = Cluster)) +
    geom_line() +
    facet_wrap(~Node, scales="free_y") +
    theme_minimal() +
    labs(title = title, y = "Value")
}



# Suppose A is your adjacency matrix (0/1, symmetric, zero diagonal)
plot_block_triangle <- function(A, z, title = "Block Graph with 3 Clusters",
                                point_radius = 0.06, seed = 1) {
  if (length(dim(A)) == 3) A <- A[1, , ]            # take first slice if 3-D
  A <- as.matrix(A)
  diag(A) <- 0
  # make undirected for plotting
  A[lower.tri(A)] <- t(A)[lower.tri(A)]
  
  g <- graph_from_adjacency_matrix(A, mode = "undirected", diag = FALSE)
  
  stopifnot(length(z) == vcount(g))
  z <- as.integer(z)
  
  # --- layout: 3 cluster centers on an equilateral triangle ---
  set.seed(seed)
  n <- length(z)
  
  # centers (roughly equilateral)
  centers <- rbind(
    c( 0,  0.90),   # top
    c(-0.9, -0.90), # bottom-left
    c( 0.9, -0.90)  # bottom-right
  )
  
  xy <- matrix(NA_real_, nrow = n, ncol = 2)
  for (k in 1:3) {
    idx <- which(z == k)
    # jitter nodes around the cluster center
    xy[idx, ] <- sweep(matrix(rnorm(2*length(idx), sd = point_radius), ncol = 2),
                       2, centers[k, ], `+`)
  }
  
  # colors and sizes
  pal <- c("skyblue2", "palegreen3", "orange")
  vcol <- pal[z]
  
  # faint edges
  ecol <- grDevices::adjustcolor("gray60", alpha.f = 0.15)
  
  plot(g,
       layout = xy,
       vertex.color = vcol,
       vertex.size = 5,
       vertex.label = NA,
       edge.color = ecol,
       edge.width = 0.5,
       main = title,
       asp = 0)
  
  invisible(list(layout = xy, colors = vcol))
}


simulate_repeat <- function(nsim = 10, 
                            cluster_sizes = c(30, 40, 50), 
                            T = 100, 
                            mus = c(0, 1, 2),
                            rho_eps = c(0.5, 0.5, 0.5),
                            burn_in = 200,
                            random_phi = FALSE) {
  adj_matrices <- array(NA, dim = c(nsim, sum(cluster_sizes), sum(cluster_sizes)))
  y <- array(NA, dim = c(nsim, sum(cluster_sizes), T))
  labels <- matrix(NA, nrow = nsim, ncol = sum(cluster_sizes))
  for (i in 1:nsim) {
    sim <- simulate_var1_randomPhi_blockSigma(T = T, cluster_sizes = cluster_sizes, 
                                              mus = mus, rho_eps = rho_eps,
                                              random_phi = random_phi, burn_in = burn_in)
    adj <- sbm_adjacency(cluster_sizes = cluster_sizes)
    adj_matrices[i, , ] <- adj$adj_matrices
    y[i, , ] <- sim$y # n by T matrix
    labels[i, ] <- sim$labels
  }
  list(adj_matrices = adj_matrices, y = y, labels = labels)
}

# Align predicted cluster labels to true labels
align_labels <- function(true, pred) {
  stopifnot(length(true) == length(pred))
  true_lv <- sort(unique(true))
  pred_lv <- sort(unique(pred))
  
  # Confusion matrix
  cm <- table(factor(true, levels = true_lv),
              factor(pred, levels = pred_lv))
  cm <- as.matrix(cm)
  
  # Pad to square
  r <- nrow(cm); c <- ncol(cm); k <- max(r, c)
  M <- matrix(0, k, k)
  M[1:r, 1:c] <- cm
  
  # Hungarian algorithm
  perm <- solve_LSAP(M, maximum = TRUE)
  rows <- as.integer(perm[1:c])
  
  # Mapping predicted → true
  map <- setNames(rep(NA, c), pred_lv)
  valid <- rows <= r
  map[valid] <- true_lv[rows[valid]]
  
  # Apply mapping
  idx <- match(pred, pred_lv)
  pred_aligned <- map[idx]
  pred_aligned <- type.convert(pred_aligned, as.is = TRUE)
  pred_aligned
}

# Compute clustering accuracy
cluster_accuracy <- function(true, pred) {
  pred_aligned <- align_labels(true, pred)
  mean(pred_aligned == true)
}




# --- example run ---
# sim <- simulate_var1_randomPhi_blockSigma()

# adj <- sbm_adjacency(cluster_sizes = c(30,40,50))



# plot_block_triangle(adj$adj_matrices, z = sim$labels,
#                     title = "Block Graph with 3 Clusters",
#                     point_radius = 0.2, seed = 1)

set.seed(112001)
data_s1_n120 <- simulate_repeat(nsim = 50, cluster_sizes = c(30,40,50), 
                                mus = c(-1, 0, 1), T = 100,
                                rho_eps = c(0.0, 0.0, 0.0), burn_in = 500,
                                random_phi = FALSE)



set.seed(312001)
data_s3_n120 <- simulate_repeat(nsim = 50, cluster_sizes = c(30,40,50), 
                                mus = c(-2, 0, 2), T = 100,
                                rho_eps = c(0.3, 0.3, 0.3), burn_in = 100,
                                random_phi = FALSE)


set.seed(3210)
data_s3_n210 <- simulate_repeat(nsim = 50, cluster_sizes = c(60,70,80), T = 100,
                                mus = c(-2, 0, 2),
                                rho_eps = c(0.3, 0.3, 0.3), burn_in = 100,
                                random_phi = FALSE)

set.seed(414401)


set.seed(419601)





plot_var_data(t(data_s3_n120$y[1,,]), z = data_s3_n120$labels[1,],
              nodes = 28:36, title = "Simulated VAR(1) Data Example (n=120)")

plot_var_data(t(data_s3_n210$y[1,,]), z = data_s3_n210$labels[1,],
              nodes = c(58, 59, 71, 100, 130, 131, 132, 180, 210), title = "Simulated VAR(1) Data Example (n=210)")



# test for cluster --------------------------------------------------------

library(NAC)

true_K <- 3

data  <- data_s3_n210$y  # seq by nodes by T
label <- data_s3_n210$labels # seq by nodes
adj   <- data_s3_n210$adj_matrices # seq by nodes by nodes

T_seq <- dim(data)[1] # seq
output_label <- array(NA, dim = dim(label)) # saved output labels

for(seq_iter in 1:T_seq){
  set.seed(42)
  
  one_adj <- adj[seq_iter,,]
  one_data <- data[seq_iter,,]
  
  output_label[seq_iter,] <- NAC(one_adj, one_data, K=true_K)
  
}

mean(sapply(1:50, function(i) cluster_accuracy(output_label[i,], label[i,])))
# mean(sapply(1:50, function(i) adjustedRandIndex(output_label[i,], label[i,])))

# ------  save data

save_as_npz <- function(filename, y, adj_matrices, labels) {
  np <- import("numpy")
  # convert R arrays -> numpy arrays
  np$savez(filename,
           y = y,   # shape: S × n × T
           adj_matrices = adj_matrices,   # shape: S × n × n
           labels = labels)   # shape: S × n
}



save_as_npz("data/data_s3_n120.npz",
            y = data_s3_n120$y,  # S x n x T
            adj_matrices = data_s3_n120$adj_matrices,         # S x n x n
            labels = data_s3_n120$labels)                # S x n

save_as_npz("data/data_s3_n210.npz",
            y = data_s3_n210$y,  # S x n x T
            adj_matrices = data_s3_n210$adj_matrices,         # S x n x n
            labels = data_s3_n210$labels)       



scenario <- "s3"  ##### "s1", "s2"
num_node <- "120" ##### s1: "120", "210", s2: "144", "196"
true_K <- 3       ##### s1: 3, s2: 4
TS_data <- np$load(paste0("data/data_",scenario,"_n",num_node,".npz"))
data  <- TS_data$f["y"]  # seq by nodes by T
label <- TS_data$f["labels"] # seq by nodes
adj   <- TS_data$f["adj_matrices"] # seq by nodes by nodes
