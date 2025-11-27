"""
Write a program to cluster a set of points using K-means for IRIS
dataset. Consider, K=3, clusters. Consider Euclidean distance as the
distance measure. Randomly initialize a cluster mean as one of the data
points. Iterate at least for 10 iterations. After iterations are over, print the
final cluster means for each of the clusters.
"""


import pandas as pd
import numpy as np

# Set random seed for reproducibility (optional)
np.random.seed(42)

# Load the dataset
# Ensure 'IRIS.csv' is in the same directory
df = pd.read_csv('IRIS.csv')

# Preprocessing:
# K-means is an unsupervised algorithm, so we drop the 'species' label column.
# We only use the numerical features (sepal_length, sepal_width, petal_length, petal_width)
X = df.drop('species', axis=1).values

print("Data Loaded.")
print(f"Shape of feature matrix: {X.shape}")
print("First 5 rows of features:\n", X[:5])

def kmeans_clustering(data, k=3, iterations=10):
    # 1. Random Initialization
    # Select 'k' random indices from the data
    random_indices = np.random.choice(data.shape[0], k, replace=False)
    
    # Set initial centroids to the data points at those indices
    centroids = data[random_indices]
    
    print("\n--- Initialization ---")
    print(f"Initial Centroids (Randomly Selected Data Points):\n{centroids}")

    # 2. Iteration Loop
    for i in range(iterations):
        # A. Calculate Distances
        # We calculate the Euclidean distance from every point to every centroid.
        # This creates a matrix of shape (n_samples, k)
        # We use numpy broadcasting: data is (150, 1, 4), centroids is (1, 3, 4)
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        
        # B. Assign Clusters
        # Find the index of the minimum distance for each data point
        cluster_labels = np.argmin(distances, axis=1)
        
        # C. Update Centroids
        new_centroids = np.zeros_like(centroids)
        for cluster_idx in range(k):
            # Get all points belonging to the current cluster
            cluster_points = data[cluster_labels == cluster_idx]
            
            # Calculate the mean of these points
            if len(cluster_points) > 0:
                new_centroids[cluster_idx] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster case (keep old centroid)
                new_centroids[cluster_idx] = centroids[cluster_idx]
        
        # Update the centroids for the next iteration
        centroids = new_centroids
        
    return centroids, cluster_labels

# Run the algorithm
k_value = 3
num_iterations = 10
final_centroids, final_labels = kmeans_clustering(X, k=k_value, iterations=num_iterations)

print("-" * 30)
print(f"Final Results after {num_iterations} iterations")
print("-" * 30)

print("Final Cluster Means (Centroids):")
# Create a DataFrame for nicer formatting
centroid_df = pd.DataFrame(final_centroids, columns=df.columns[:-1])
centroid_df.index.name = 'Cluster'
print(centroid_df)

# Optional: Show how many points ended up in each cluster
print("\nPoints per Cluster:")
unique, counts = np.unique(final_labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} points")