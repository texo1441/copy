"""
Write a program to do the following: You have given a collection of 8
points. P1=[2, 10] P2=[2, 5] P3=[8, 4] P4=[5, 8] P5=[7,5] P6=[6, 4] P7=[1, 2]
P8=[4, 9]. Perform the k-mean clustering with initial centroids as m1=P1
=Cluster#1=C1 and m2=P4=cluster#2=C2, m3=P7 =Cluster#3=C3. Answer
the following 1] Which cluster does P6 belong to? 2] What is the
population of a cluster around m3? 3] What is the updated value of m1,
m2, m3?
"""


import numpy as np
import pandas as pd

# Using numpy for efficient distance calculations

# 1. Define the points
points = {
    'P1': np.array([2, 10]),
    'P2': np.array([2, 5]),
    'P3': np.array([8, 4]),
    'P4': np.array([5, 8]),
    'P5': np.array([7, 5]),
    'P6': np.array([6, 4]),
    'P7': np.array([1, 2]),
    'P8': np.array([4, 9])
}

# 2. Define Initial Centroids
# m1 = P1 (Cluster 1)
# m2 = P4 (Cluster 2)
# m3 = P7 (Cluster 3)
centroids = {
    'C1': points['P1'],
    'C2': points['P4'],
    'C3': points['P7']
}

print("Initial Centroids:")
for name, val in centroids.items():
    print(f"{name}: {val}")

# Initialize dictionaries to hold points for each cluster
clusters = {
    'C1': [],
    'C2': [],
    'C3': []
}
cluster_labels = {
    'C1': [],
    'C2': [],
    'C3': []
}

# Helper function to calculate Euclidean distance
def calculate_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Loop through each point to find the nearest centroid
print("--- Distance Calculations ---")
for p_name, p_coords in points.items():
    # Calculate distance to each centroid
    dist_c1 = calculate_distance(p_coords, centroids['C1'])
    dist_c2 = calculate_distance(p_coords, centroids['C2'])
    dist_c3 = calculate_distance(p_coords, centroids['C3'])
    
    # Find the minimum distance
    distances = {'C1': dist_c1, 'C2': dist_c2, 'C3': dist_c3}
    closest_cluster = min(distances, key=distances.get)
    
    # Assign point to the closest cluster
    clusters[closest_cluster].append(p_coords)
    cluster_labels[closest_cluster].append(p_name)
    
    print(f"{p_name}: Dist to C1={dist_c1:.2f}, C2={dist_c2:.2f}, C3={dist_c3:.2f} -> Assigned to {closest_cluster}")

print("\n--- Cluster Membership ---")
for c_name, labels in cluster_labels.items():
    print(f"{c_name}: {labels}")

# --- Question 1: Which cluster does P6 belong to? ---
p6_cluster = None
for c_name, labels in cluster_labels.items():
    if 'P6' in labels:
        p6_cluster = c_name
        break
print(f"1] P6 belongs to: {p6_cluster}")

# --- Question 2: What is the population of a cluster around m3 (C3)? ---
pop_m3 = len(clusters['C3'])
print(f"2] Population of cluster around m3: {pop_m3}")

# --- Question 3: Updated value of m1, m2, m3 ---
new_centroids = {}
print("3] Updated values of centroids:")

for c_name, p_list in clusters.items():
    if p_list: # Check if cluster is not empty
        # Calculate mean of all points in the cluster
        new_mean = np.mean(p_list, axis=0)
        new_centroids[c_name] = new_mean
        print(f"   Updated {c_name} (m{c_name[-1]}): {new_mean}")
    else:
        print(f"   Updated {c_name}: No points assigned (remains same)")