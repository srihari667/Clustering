# Clustering Assignment

This repository contains the Clustering assignment covering **unsupervised learning**, including **K-Means, Hierarchical Clustering, DBSCAN**, and practical exercises using synthetic and real datasets. All questions are listed with continuous serial numbers and marked as **Theory** or **Practical**.

---

## Table of Contents
1. [All Questions](#all-questions)
2. [Python Libraries Used](#python-libraries-used)

---

## All Questions

| S.No | Type | Question / Task |
|------|------|----------------|
| 1 | Theory | What is unsupervised learning in the context of machine learning? |
| 2 | Theory | How does K-Means clustering algorithm work? |
| 3 | Theory | Explain the concept of a dendrogram in hierarchical clustering. |
| 4 | Theory | What is the main difference between K-Means and Hierarchical Clustering? |
| 5 | Theory | What are the advantages of DBSCAN over K-Means? |
| 6 | Theory | When would you use Silhouette Score in clustering? |
| 7 | Theory | What are the limitations of Hierarchical Clustering? |
| 8 | Theory | Why is feature scaling important in clustering algorithms like K-Means? |
| 9 | Theory | How does DBSCAN identify noise points? |
| 10 | Theory | Define inertia in the context of K-Means. |
| 11 | Theory | What is the elbow method in K-Means clustering? |
| 12 | Theory | Describe the concept of "density" in DBSCAN. |
| 13 | Theory | Can hierarchical clustering be used on categorical data? |
| 14 | Theory | What does a negative Silhouette Score indicate? |
| 15 | Theory | Explain the term "linkage criteria" in hierarchical clustering. |
| 16 | Theory | Why might K-Means clustering perform poorly on data with varying cluster sizes or densities? |
| 17 | Theory | What are the core parameters in DBSCAN, and how do they influence clustering? |
| 18 | Theory | How does K-Means++ improve upon standard K-Means initialization? |
| 19 | Theory | What is agglomerative clustering? |
| 20 | Theory | What makes Silhouette Score a better metric than just inertia for model evaluation? |
| 21 | Practical | Generate synthetic data with 4 centers using `make_blobs` and apply K-Means clustering. Visualize using a scatter plot. |
| 22 | Practical | Load the Iris dataset and use Agglomerative Clustering to group the data into 3 clusters. Display the first 10 predicted labels. |
| 23 | Practical | Generate synthetic data using `make_moons` and apply DBSCAN. Highlight outliers in the plot. |
| 24 | Practical | Load the Wine dataset and apply K-Means clustering after standardizing the features. Print the size of each cluster. |
| 25 | Practical | Use `make_circles` to generate synthetic data and cluster it using DBSCAN. Plot the result. |
| 26 | Practical | Load the Breast Cancer dataset, apply MinMaxScaler, and use K-Means with 2 clusters. Output the cluster centroids. |
| 27 | Practical | Generate synthetic data using `make_blobs` with varying cluster standard deviations and cluster with DBSCAN. |
| 28 | Practical | Load the Digits dataset, reduce it to 2D using PCA, and visualize clusters from K-Means. |
| 29 | Practical | Create synthetic data using `make_blobs` and evaluate silhouette scores for k = 2 to 5. Display as a bar chart. |
| 30 | Practical | Load the Iris dataset and use hierarchical clustering to group data. Plot a dendrogram with average linkage. |
| 31 | Practical | Generate synthetic data with overlapping clusters using `make_blobs`, then apply K-Means and visualize with decision boundaries. |
| 32 | Practical | Load the Digits dataset and apply DBSCAN after reducing dimensions with t-SNE. Visualize the results. |
| 33 | Practical | Generate synthetic data using `make_blobs` and apply Agglomerative Clustering with complete linkage. Plot the result. |
| 34 | Practical | Load the Breast Cancer dataset and compare inertia values for K = 2 to 6 using K-Means. Show results in a line plot. |
| 35 | Practical | Generate synthetic concentric circles using `make_circles` and cluster using Agglomerative Clustering with single linkage. |
| 36 | Practical | Use the Wine dataset, apply DBSCAN after scaling the data, and count the number of clusters (excluding noise). |
| 37 | Practical | Generate synthetic data with `make_blobs` and apply KMeans. Then plot the cluster centers on top of the data points. |
| 38 | Practical | Load the Iris dataset, cluster with DBSCAN, and print how many samples were identified as noise. |
| 39 | Practical | Generate synthetic non-linearly separable data using `make_moons`, apply K-Means, and visualize the clustering result. |
| 40 | Practical | Load the Digits dataset, apply PCA to reduce to 3 components, then use KMeans and visualize with a 3D scatter plot. |
| 41 | Practical | Generate synthetic blobs with 5 centers and apply KMeans. Then use `silhouette_score` to evaluate the clustering. |
| 42 | Practical | Load the Breast Cancer dataset, reduce dimensionality using PCA, and apply Agglomerative Clustering. Visualize in 2D. |
| 43 | Practical | Generate noisy circular data using `make_circles` and visualize clustering results from KMeans and DBSCAN side-by-side. |
| 44 | Practical | Load the Iris dataset and plot the Silhouette Coefficient for each sample after KMeans clustering. |
| 45 | Practical | Generate synthetic data using `make_blobs` and apply Agglomerative Clustering with 'average' linkage. Visualize clusters. |
| 46 | Practical | Load the Wine dataset, apply KMeans, and visualize the cluster assignments in a seaborn pairplot (first 4 features). |
| 47 | Practical | Generate noisy blobs using `make_blobs` and use DBSCAN to identify both clusters and noise points. Print the count. |
| 48 | Practical | Load the Digits dataset, reduce dimensions using t-SNE, then apply Agglomerative Clustering and plot the clusters. |

---

## Python Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
