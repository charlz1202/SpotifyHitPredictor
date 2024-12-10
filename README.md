# Spotify Hit Predictor

## Project Description
The Spotify Hit Predictor project uses clustering algorithms and data analytics to explore patterns in Spotify's dataset. The primary goal of this project is to uncover meaningful insights about music trends and features that may influence a song's success. By applying various clustering techniques and dimensionality reduction methods, this project provides a comprehensive analysis of Spotify's data.

## Table of Contents
1. [Loading the Data and Feature Engineering](#loading-the-data-and-feature-engineering)
    - Importing Required Packages
    - Loading the Dataset
    - Exploratory Data Analysis
    - Data Cleaning
    - Feature Engineering
2. [Clustering Techniques](#clustering-techniques)
    - [K-Means](#k-means)
        - Finding k (clusters) to use
        - Applying K-Means Algorithm
    - [PAMK (Partitioning Around Medoids)](#pamk)
        - Automatic Selection of k
    - [DBSCAN (Density-Based Spatial Clustering)](#dbscan)
        - Without PCA
        - Using PCA
    - [Hierarchical Clustering](#hierarchical-clustering)
        - Complete Linkage
        - Single Linkage
3. [Conclusion](#conclusion)

---

## 1. Loading the Data and Feature Engineering

### Importing Required Packages
This project utilized the following R libraries for data manipulation, visualization, and clustering:
- `dplyr`
- `ggplot2`
- `cluster`
- `NbClust`
- `factoextra`
- `dbscan`

### Loading the Dataset
The Spotify dataset was loaded into R for analysis. Key features include:
- Tempo
- Energy
- Danceability
- Key
- Mode
- Time Signature

### Exploratory Data Analysis (EDA)
Exploratory analysis was performed to understand the structure and distribution of the dataset. This included visualizing data distributions and identifying potential outliers.

### Data Cleaning
Data cleaning included:
- Handling missing values
- Removing duplicates
- Standardizing column names

### Feature Engineering
- **Summary of each variable:** A detailed summary of each column.
- **Boxplot and Histogram:** Visualized data distributions for all columns.
- **One-Hot Encoding:** Applied to categorical variables such as `key`, `mode`, and `time_signature`.
- **Normalization:** Numeric variables were normalized for clustering.

---

## 2. Clustering Techniques

### K-Means
#### Finding k (clusters) to use:
- **Elbow Method:** Used to identify the optimal number of clusters.
- **NbClust:** Analyzed multiple criteria to determine the best k.

#### Applying K-Means Algorithm:
- Performed clustering with 4 and 5 clusters to compare results.

### PAMK (Partitioning Around Medoids)
- Automatic selection of k.
- Clustering with k = 5 based on K-Means results.

### DBSCAN (Density-Based Spatial Clustering)
#### Without PCA:
- Applied DBSCAN directly to the dataset.

#### Using PCA:
- Dimensionality reduction via PCA before applying DBSCAN.

### Hierarchical Clustering
#### Complete Linkage:
- Clustering performed with the "complete" linkage method.

#### Single Linkage:
- Clustering performed with the "single" linkage method.

---

## 3. Conclusion
The Spotify Hit Predictor project successfully demonstrated the use of clustering techniques to identify meaningful groupings in Spotify's dataset. Key takeaways include:
- K-Means and PAMK effectively segmented the dataset into optimal clusters.
- DBSCAN was particularly useful in identifying noise and outliers.
- PCA enhanced the clustering process for DBSCAN.
- Hierarchical clustering provided additional insights into the structure of the data.

This project highlights the power of clustering and feature engineering in uncovering hidden patterns within a dataset.

---

## Repository Structure
```
SpotifyHitPredictor/
|-- data/               # Contains the Spotify dataset
|-- scripts/            # R scripts for analysis and clustering
|-- figures/            # Visualizations and plots
|-- README.md           # Project documentation
```

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/charlz1202/spotifyhitpredictor.git
   ```
2. Open the R project file in your RStudio.
3. Run the scripts in the `scripts/` folder to reproduce the analysis and clustering.

---

## Technologies Used
- R Programming
- Libraries: `dplyr`, `ggplot2`, `cluster`, `NbClust`, `factoextra`, `dbscan`

---

## Acknowledgments
Special thanks to the creators of the Spotify dataset and the R community for providing powerful tools and resources that made this project possible.

---

## Connect
Feel free to connect with me to discuss this project or collaborate on future data analytics initiatives:
- GitHub: [charlz1202](https://github.com/charlz1202/spotifyhitpredictor)
- LinkedIn: [Charlie Medialdea](https://www.linkedin.com/in/charliemedialdea/)

