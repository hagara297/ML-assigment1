#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons


# In[20]:


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# In[21]:


def pearson_correlation_distance(point1, point2):
    mean1 = np.mean(point1)
    mean2 = np.mean(point2)
    std1 = np.std(point1)
    std2 = np.std(point2)
    correlation = np.sum((point1 - mean1) * (point2 - mean2)) / (len(point1) * std1 * std2)
    return 1 - correlation


# In[22]:


def GUC_Distance(Cluster_Centroids, Data_points, Distance_Type='euclidean'):
    if Distance_Type == 'euclidean':
        distance_func = euclidean_distance
    elif Distance_Type == 'pearson':
        distance_func = pearson_correlation_distance
    else:
        raise ValueError("Invalid distance type. Supported types: 'euclidean', 'pearson'")
    
    distances = np.zeros((Data_points.shape[0], Cluster_Centroids.shape[0]))
    for i, centroid in enumerate(Cluster_Centroids):
        for j, point in enumerate(Data_points):
            distances[j, i] = distance_func(centroid, point)
    
    return distances


# In[23]:


def GUC_Kmean(Data_points, Number_of_Clusters, Distance_Type, max_iterations=100, tolerance=1e-4):
    #  Initialize cluster centroids randomly
    min_vals = np.min(Data_points, axis=0)
    max_vals = np.max(Data_points, axis=0)
    Cluster_Centroids = np.random.uniform(min_vals, max_vals, size=(Number_of_Clusters, Data_points.shape[1]))
    
    iteration = 0
    prev_cluster_distances = None
    
    while iteration < max_iterations:
        #  Cluster Assignment
        Cluster_Distances = GUC_Distance(Cluster_Centroids, Data_points, Distance_Type)
        Cluster_Assignments = np.argmin(Cluster_Distances, axis=1)
        
        #  Calculate mean square distance for each cluster
        cluster_distances = np.zeros(Number_of_Clusters)
        for i in range(Number_of_Clusters):
            cluster_distances[i] = np.mean(Cluster_Distances[Cluster_Assignments == i, i])
        
        #  Update centroids
        new_cluster_centroids = np.array([np.mean(Data_points[Cluster_Assignments == i], axis=0) for i in range(Number_of_Clusters)])
        
        #  Calculate cluster metric (sum of squared error)
        cluster_metric = np.sum(cluster_distances)
        
        #  Stopping condition
        if prev_cluster_distances is not None and np.abs(prev_cluster_distances - cluster_metric) < tolerance:
            break
        
        Cluster_Centroids = new_cluster_centroids
        prev_cluster_distances = cluster_metric
        iteration += 1
    
    return Cluster_Distances, cluster_metric

#testt
Data_points = np.array([[1, 2], [2, 3], [6, 8], [7, 9], [10, 12]])
Number_of_Clusters = 2
Distance_Type = 'euclidean'

Final_Cluster_Distance, Cluster_Metric = GUC_Kmean(Data_points, Number_of_Clusters, Distance_Type)
print("Final cluster distances:\n", Final_Cluster_Distance)
print("Cluster metric:", Cluster_Metric)


# In[24]:



from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the display_cluster function
def display_cluster(X, km=[], num_clusters=0):
    color = 'brgcmyk'  # List colors
    alpha = 0.5  # Color opacity
    s = 20

    # Determine the number of dimensions
    num_dimensions = X.shape[1]

    # Prepare the figure size and background
    plt.rcParams['figure.figsize'] = [12, 6]
    sns.set_style("whitegrid")
    sns.set_context("talk")

    if num_dimensions == 2:
        if num_clusters == 0:
            plt.scatter(X[:, 0], X[:, 1], c=color[0], alpha=alpha, s=s)
        else:
            for j in range(num_clusters):
                plt.scatter(X[km.labels_ == j, 0], X[km.labels_ == j, 1], c=color[j], alpha=alpha, s=s)
                plt.scatter(km.cluster_centers_[j][0], km.cluster_centers_[j][1], c=color[j], marker='x', s=100)
    else:
        print("Number of dimensions is not 2. Unable to display clusters.")

# Example 1: Circular Data Gen and display
plt.figure(figsize=[12, 6])
plt.subplot(2, 1, 1)
angle = np.linspace(0, 2*np.pi, 100, endpoint=False)
X_circle = np.append([np.cos(angle)], [np.sin(angle)], 0).transpose()
display_cluster(X_circle)

# Example 3: moons Data Gen and display
n_samples = 1000
plt.subplot(2, 1, 2)
X_moons, _ = make_moons(n_samples=n_samples, noise=0.1)
display_cluster(X_moons)


# Testing the GUC_Kmean function on Simple 2D
num_clusters_range = range(2, 11)
cluster_metrics_circle = []
cluster_metrics_moons = []

for num_clusters in num_clusters_range:
    # Circular data
    kmeans_circle = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_circle.fit(X_circle)
    cluster_metrics_circle.append(kmeans_circle.inertia_)

    # Moons data
    kmeans_moons = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_moons.fit(X_moons)
    cluster_metrics_moons.append(kmeans_moons.inertia_)

# Plotting Cluster performance Metric versus the number of clusters
plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(num_clusters_range, cluster_metrics_circle, marker='o')
plt.title('Circular Data: Cluster Performance Metric')
plt.xlabel('Number of Clusters')
plt.ylabel('Cluster Metric')

plt.subplot(1, 2, 2)
plt.plot(num_clusters_range, cluster_metrics_moons, marker='o')
plt.title('Moons Data: Cluster Performance Metric')
plt.xlabel('Number of Clusters')
plt.ylabel('Cluster Metric')

plt.tight_layout()
plt.show()


# In[25]:


customer_data = pd.read_csv('Mall_Customers.csv')


print(customer_data.head())

# Extract relevant features for clustering
features = customer_data[['Age', 'Annual_Income_(k$)', 'Spending_Score']]

# Define the range of clusters to try
num_clusters_range = range(2, 11)

# Initialize lists to store cluster metrics for each number of clusters
cluster_metrics_euclidean = []
cluster_metrics_pearson = []

# Apply GUC_Kmean function with varying number of clusters
for num_clusters in num_clusters_range:
    # Euclidean distance
    kmeans_euclidean = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_euclidean.fit(features)
    cluster_metrics_euclidean.append(kmeans_euclidean.inertia_)

    # Pearson correlation distance
    kmeans_pearson = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_pearson.fit(features)
    cluster_metrics_pearson.append(kmeans_pearson.inertia_)

# Plot the cluster metric versus the number of clusters
plt.figure(figsize=[12, 6])
plt.plot(num_clusters_range, cluster_metrics_euclidean, marker='o', label='Euclidean Distance')
plt.plot(num_clusters_range, cluster_metrics_pearson, marker='o', label='Pearson Correlation Distance')
plt.title('Cluster Metric versus Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Cluster Metric')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




