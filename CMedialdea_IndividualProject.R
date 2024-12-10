#CSIS 4260-002
#Individual Project
#Submitted By: Charlie Medialdea
#Dataset: Spotify Hit Predictor Dataset


# Import required packages
  library(psych)
  library(dplyr)
  library(NbClust)
  library(parallel)
  library(ggplot2)
  library(fpc)
  library(cluster)
  library(dbscan)
  library(tidyverse)
  library(factoextra)
  
# Loading the Data set
  spotify_data <- read.csv("C:\\Users\\charl\\OneDrive - Douglas College\\Special Topics in Data Analytics\\Individual Project\\SpotifyHitPredictor.csv", header = TRUE)

  
# Exploratory Data Analysis
  
  # Get the number of rows and columns.
  dim(spotify_data)

  # List the names of all variables in the data set.
  colnames(spotify_data)
  
  # Check the structure and types of each column.
  str(spotify_data)


  # See the first few rows of the dataset.
  head(spotify_data)
  
  
# Data Cleaning

  # Check the missing values: 
    colSums(is.na(spotify_data))
  
  # Dropping the column 'track', 'artist', 'uri' as this is non-numerical and does not have impact to my model. Also,           removing the target variable.
    spotify_data <- select(spotify_data, -track, -artist, -uri, -target)
  

# Feature Engineering

  # Checking the data distribution is more important than checking data imbalance in clustering since this is unsupervised      learning.
    
    # Get a quick summary of each variable
    summary(spotify_data)
    
    
  ## Visualization
    # BOXPLOTS
    
    # List of numeric columns to create boxplots
    numeric_columns <- c("danceability", "energy", "key", "loudness", "mode", 
                         "speechiness", "acousticness", "instrumentalness", "liveness", 
                         "valence", "tempo", "duration_ms", "time_signature", "chorus_hit","sections")
    
    # Set up a 5x3 plot grid
    par(mfrow=c(5, 3), mar=c(4, 4, 2, 1))  
    
    # Create the boxplot for each column
    for (col in numeric_columns) {
      boxplot(spotify_data[[col]], main=paste("Outlier Detection for", col))
    }
    
    
    # HISTOGRAMS
    # List of numeric columns to create histogram
    numeric_columns <- c("danceability", "energy", "key", "loudness", "mode", 
                         "speechiness", "acousticness", "instrumentalness", "liveness", 
                         "valence", "tempo", "duration_ms", "time_signature", "chorus_hit","sections")
    
    # Set up a 5x3 plot grid
    par(mfrow=c(5, 3), mar=c(4, 4, 2, 1))  
    
    # Create the histogram for each column
    for (col in numeric_columns) {
      hist(spotify_data[[col]], main=paste("Distribution of", col), xlab=col)
    }
    
    
    # Converting the categorical column key and time_signature using one-hot encoding
      key_encoded <- model.matrix(~ key - 1, data = spotify_data)
      time_signature_encoded <- model.matrix(~ time_signature - 1, data = spotify_data)
      mode_encoded <- model.matrix(~ mode - 1, data = spotify_data)
    
      key_encoded_df <- as.data.frame(key_encoded)
      time_signature_encoded_df <- as.data.frame(time_signature_encoded)
      mode_encoded_df <- as.data.frame(mode_encoded)

    # Dropping the original 'key' & 'time_signature' columns
      spotify_data <- select(spotify_data, -time_signature, -key, -mode)
    
    
    # Adding back the converted columns to spotify_data
    spotify_data <- cbind(spotify_data, key_encoded_df, time_signature_encoded_df, mode_encoded_df)
   
     
    # Normalize the numeric variables
    numeric_columns <- c('danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                         'instrumentalness', 'liveness', 'valence', 'tempo', 'chorus_hit')
    
    # Apply Z-score standardization to numeric data
    spotify_data[numeric_columns] <- scale(spotify_data[numeric_columns])

    str(spotify_data)  
    
# K-Means Clustering

      # Finding the correct 'k' by using wss (elbow mehtod). 
      dev.new(width=10, height=8)
    
      wss <- (nrow(spotify_data)-1)*sum(apply(spotify_data, 2, var))
      for (i in 2:15) wss[i] <- sum(kmeans(spotify_data, centers=i)$tot.withinss)
    
      # Plot WSS to find the optimal K
      plot(1:15, wss, type="b", col="red", xlab="Number of Clusters (K)", ylab="Within-Cluster Sum of Squares (WSS)",
      main="Elbow Method for Optimal K")
    
    
      # Using NbClust to find the optimal number of clusters
      nc <- NbClust(spotify_data, min.nc = 2, max.nc = 6, method = "kmeans", index = "ch")
     
      # Create and display the table of suggested cluster numbers
      Bar <- table(nc$Best.partition)
      Bar
      
      barplot(Bar, main = "Optimal Number of Clusters", 
              xlab = "Number of Clusters", ylab = "Frequency", col = "lightblue", las = 1)
      
      
    # Applying K-Means clustering using k=4 as result of wss and NBClust
      kmeans1 <- kmeans(spotify_data, 4)
      kmeans1
      
    # Using Principal Component Analysis (PCA) to reduce correlation and dimensionality since our dataset  is       with 15       columns, making it easier to visualize the clusters in a lower-dimensional space such as 2D.
      
      pca_result <- prcomp(spotify_data, center = TRUE, scale. = TRUE)
      pca_data <- data.frame(pca_result$x[, 1:2])
      pca_data$Cluster <- as.factor(kmeans1$cluster)
      
    # Create a 2D scatter plot of the clusters
      centroids <- pca_data %>%
      group_by(Cluster) %>%
      summarize(PC1 = mean(PC1), PC2 = mean(PC2))
      
      ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
      geom_point(alpha = 0.7, size = 3) +
      geom_point(data = centroids, aes(x = PC1, y = PC2), color = "black", shape = "x", size = 6, stroke = 2) +
      labs(title = "K-Means Clusters (PCA)", x = "Principal Component 1", y = "Principal Component 2") +
      scale_color_manual(values = rainbow(length(unique(pca_data$Cluster)))) +
      theme_minimal()
      
    # Applying K-Means clustering using k=5
      kmeans5 <- kmeans(spotify_data, 5)
      kmeans5
      
    # Here I am performing Principal Component Analysis (PCA) to reduce correlation and dimensionality since our dataset  is       with 15 columns, making it easier to visualize the clusters in a lower-dimensional space such as 2D.
      pca_result <- prcomp(spotify_data, center = TRUE, scale. = TRUE)
      pca_data <- data.frame(pca_result$x[, 1:2])
      pca_data$Cluster <- as.factor(kmeans5$cluster)
      
    # Create a 2D scatter plot of the clusters
      centroids <- pca_data %>%
      group_by(Cluster) %>%
      summarize(PC1 = mean(PC1), PC2 = mean(PC2))
      
      ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
      geom_point(alpha = 0.7, size = 3) +
      geom_point(data = centroids, aes(x = PC1, y = PC2), color = "black", shape = "x", size = 6, stroke = 2) +
      labs(title = "K-Means Clusters (PCA)", x = "Principal Component 1", y = "Principal Component 2") +
      scale_color_manual(values = rainbow(length(unique(pca_data$Cluster)))) +
      theme_minimal()
      
      
        
# PAMK (Partitioning Around Medoids)
    # Performing Principal Component Analysis (PCA) to reduce correlation and dimensionality of the dataset with 15 columns       important when doing PAMK
      pca_result <- prcomp(spotify_data, center = TRUE, scale. = TRUE)
      pca_data <- data.frame(pca_result$x[, 1:2])
      
    # Making sure that pca_data only contains numeric columns for clustering
      pca_data_numeric <- pca_data[, c("PC1", "PC2")]
  
    # Ensure the data is numeric matrix
      pca_data_numeric_matrix <- as.matrix(pca_data_numeric)
      
    # Perform PAMK clustering with automatic selection of K.
      pam_result <- pamk(pca_data_numeric_matrix)

    # Get the optimal number of clusters
      optimal_clusters <- pam_result$nc
      cat("Optimal number of clusters:", optimal_clusters, "\n")
      
    # Check the clustering results
      pam_result$pamobject  
      
    # Check the distribution of clusters in pam_result
      table(pam_result$pamobject$clustering)
      
       
    # Set up the layout for two side-by-side plots with smaller margins
      par(mfrow = c(1, 2), mar = c(6, 6, 5, 2)) 
      
    # Calculate silhouette values for the PAM clustering result
      sil <- silhouette(pam_result$pamobject$clustering, dist(pca_data_numeric))
      
    # Plot the cluster result on the right using the first two components (PCA)
      clusplot(pca_data_numeric, pam_result$pamobject$clustering, 
      color=TRUE, shade=TRUE, labels=2, lines=0, 
      main="PAM Clustering with Optimal K", 
      cex.main=0.8, cex.lab=0.8, cex.axis=0.7) 
      
    # Plot the silhouette on the left
      plot(sil, main = "Silhouette Plot for PAM Clustering", 
      col = 2:(max(pam_result$pamobject$clustering) + 1), 
      border = NA, 
      cex.main=0.8, cex.lab=0.8, cex.axis=0.7)  
    

    # Perform PAMK clustering using K=4
      pca_result <- prcomp(spotify_data, center = TRUE, scale. = TRUE)
      pca_data <- data.frame(pca_result$x[, 1:2])
      pca_data_numeric <- pca_data[, c("PC1", "PC2")]
      pca_data_numeric_matrix <- as.matrix(pca_data_numeric)
      pam_result <- pamk(pca_data_numeric_matrix, k=4)
    # Check the clustering results
      pam_result$pamobject  
      
    # Check the distribution of clusters in pam_result
      table(pam_result$pamobject$clustering)
      
    # Set up the layout for two side-by-side plots with smaller margins
      par(mfrow = c(1, 2), mar = c(6, 6, 5, 2)) 
      
    # Calculate silhouette values for the PAM clustering result
      sil <- silhouette(pam_result$pamobject$clustering, dist(pca_data_numeric))
      
    # Plot the cluster result on the right using the first two components (PCA)
      clusplot(pca_data_numeric, pam_result$pamobject$clustering, 
               color=TRUE, shade=TRUE, labels=2, lines=0, 
               main="PAM Clustering with Optimal K", 
               cex.main=0.8, cex.lab=0.8, cex.axis=0.7) 
      
    # Plot the silhouette on the left
      plot(sil, main = "Silhouette Plot for PAM Clustering", 
           col = 2:(max(pam_result$pamobject$clustering) + 1), 
           border = NA, 
           cex.main=0.8, cex.lab=0.8, cex.axis=0.7)  
      


# DBSCAN
      
      # Plot the k-NN distance to estimate eps
      kNNdistplot(spotify_data, k = ncol(spotify_data) + 1)
      abline(h = 0.5, col = "green")  # Initial guide for eps
      
      # Function to calculate optimal eps
      eps <- function(mydata) {
        dist <- dbscan::kNNdist(mydata, ncol(mydata) + 1)
        dist <- dist[order(dist)]
        dist <- dist / max(dist)
        ddist <- diff(dist) / (1 / length(dist))
        EPS <- dist[length(ddist) - length(ddist[ddist > 1])]
        print(EPS)
        return(EPS)
      }
      
      # Calculate and print initial eps
      initial_eps <- eps(spotify_data)
      abline(h = initial_eps, col = "red", lty = 5)
      
      # Perform DBSCAN with the calculated eps
      db1 <- dbscan(spotify_data, eps = initial_eps, minPts = ncol(spotify_data) + 1)
      db1
      
      # To fix or enhance the result of my initial eps and DBSCAN result, I will be using PCA to reduce the dimension of my dataset to           try              enhance the result of my DBSCAN.
      
      # Perform PCA on the spotify_data
      pca_result <- prcomp(spotify_data, center = TRUE, scale. = TRUE)
      
      # Reduce data to the first two principal components
      pca_data <- data.frame(pca_result$x[, 1:2])
      
      # Plot the k-NN distance to estimate eps using the PCA-reduced data
      kNNdistplot(pca_data, k = ncol(pca_data) + 1)
      abline(h = 0.5, col = "green")  # Initial guide for eps
      
      # Function to calculate optimal eps
      eps <- function(mydata) {
        dist <- dbscan::kNNdist(mydata, ncol(mydata) + 1)
        dist <- dist[order(dist)]
        dist <- dist / max(dist)
        ddist <- diff(dist) / (1 / length(dist))
        EPS <- dist[length(ddist) - length(ddist[ddist > 1])]
        print(EPS)
        return(EPS)
      }
      
      # Calculate and print initial eps for PCA-reduced data
      initial_eps <- eps(pca_data)
      abline(h = initial_eps, col = "red", lty = 5)
      
      # Perform DBSCAN with the calculated eps on PCA-reduced data
      db1 <- dbscan(pca_data, eps = initial_eps, minPts = ncol(pca_data) + 1)
      print(db1)
   
      # Plot initial DBSCAN clustering result using PCA-reduced data
      hullplot(pca_data, db1)
      plot(pca_data, col = db1$cluster, main = "DBSCAN Clustering with Initial Eps (PCA-reduced Data)", xlab = "PC1", ylab = "PC2")
      
      # Identify noise points (cluster 0) and remove them from PCA-reduced data
      noise <- which(db1$cluster == 0)
      noise 
      
      # Create a cleaned dataset without noise (PCA-reduced data)
      pca_data_clean <- pca_data[-noise, ]
      
      # Recalculate eps on the cleaned dataset
      new_eps <- eps(pca_data_clean)
      abline(h = new_eps, col = "blue", lty = 5)
      
      # Perform DBSCAN again on the cleaned dataset with the new eps
      db2 <- dbscan(pca_data_clean, eps = new_eps, minPts = ncol(pca_data_clean) + 1)
      db2
      
      # Plot the final clustering result
      hullplot(pca_data_clean, db2)
      plot(pca_data_clean, col = db2$cluster, main = "DBSCAN Clustering After Removing Noise (PCA-reduced Data)", xlab = "PC1", ylab = "PC2")


# Hierarchical Clustering
    
      data2 <- scale(spotify_data)
      data3 <- as.data.frame(data2)
      
      d <- dist(spotify_data, method = "euclidean")
      hc1 <- hclust(d, method = "complete")
      hc1
      par(mar=c(6, 6, 3, 3))
      plot(hc1, cex=0.6)
      
      #Cut into k groups
      rect.hclust(hc1, k=4, border=1.4)
      
      cutree(hc1, k=4)
      fviz_cluster(list(data=data3, cluster=cutree(hc1, k=4)), labelsize = 7)
      
      # Add cluster labels to the data
      data4 <- mutate(data3, group=cutree(hc1, k=4))
      head(data4)
      
      
      
      hc2 <- hclust(d, method = "single")
      hc2
      plot(hc2, cex=0.6)
      
      #Cut into k groups
      rect.hclust(hc2, k=6, border=1.4)
      cutree(hc2, k=6)
      fviz_cluster(list(data=data3, cluster=cutree(hc2, k=6)), labelsize = 7)
      