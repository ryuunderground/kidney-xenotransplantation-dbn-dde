### Cluster number diagnostics (elbow + silhouette)
### Repeat 3 times, changing the sample folder/output names: short, inter, long

library(tidyverse)
library(cluster)
library(factoextra)
library(ggplot2)
library(patchwork)

root <- "~/Documents/github/kidney-xenotransplantation-dbn-dde/preprocessing/DBN-DDE_inputs"
setwd(root)
getwd()

# Load data
df <- read.csv("short_input.csv", row.names = 1)

# Gene-wise scaling, drop rows with NA after scaling
df_scaled <- t(scale(t(df)))
df_scaled <- df_scaled[complete.cases(df_scaled), ]

# -------------------------
# Elbow method
# -------------------------
set.seed(123)
max_k <- 10
wss <- numeric(max_k)
for (k in 1:max_k) {
  km <- kmeans(df_scaled, centers = k, nstart = 25)
  wss[k] <- km$tot.withinss
}

elbow_df <- data.frame(
  k = 1:max_k,
  WSS = wss
)

p1 <- ggplot(elbow_df, aes(x = k, y = WSS)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 1:max_k) +  # integer ticks only
  theme_bw(base_size = 14) +
  labs(title = "Elbow Method",
       x = "Number of clusters (K)",
       y = "Total within-cluster sum of squares")

ggsave("../k-means_diagnosis/short_elbow.pdf", p1, width = 6, height = 5)

# -------------------------
# Silhouette method
# -------------------------
sil_width <- numeric(max_k)
for (k in 2:max_k) {
  km <- kmeans(df_scaled, centers = k, nstart = 25)
  ss <- silhouette(km$cluster, dist(df_scaled))
  sil_width[k] <- mean(ss[, 3])
}

sil_df <- data.frame(
  k = 2:max_k,
  Silhouette = sil_width[2:max_k]
)

p2 <- ggplot(sil_df, aes(x = k, y = Silhouette)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 2:max_k) +  # integer ticks only
  theme_bw(base_size = 14) +
  labs(title = "Silhouette Method",
       x = "Number of clusters (K)",
       y = "Average silhouette width")

ggsave("../k-means_diagnosis/short_silhouette.pdf", p2, width = 6, height = 5)

# -------------------------
# Combined plot
# -------------------------
p_combined <- p1 + p2
ggsave("../k-means_diagnosis/short_cluster_diagnostics.pdf", p_combined, width = 10, height = 5)

