"""
Write a program to do the following: You have given a collection of 8
points. P1=[0.1,0.6] P2=[0.15,0.71] P3=[0.08,0.9] P4=[0.16, 0.85]
P5=[0.2,0.3] P6=[0.25,0.5] P7=[0.24,0.1] P8=[0.3,0.2]. Perform the k-mean
clustering with initial centroids as m1=P1 =Cluster#1=C1 and
m2=P8=cluster#2=C2. Answer the following 1] Which cluster does P6
belong to? 2] What is the population of a cluster around m2? 3] What is
the updated value of m1 and m2?
"""


import numpy as np
import pandas as pd

# We use numpy for efficient mathematical operations (distance calculation)

# 1. Define the points
# using a dictionary for easy labeling (P1, P2...)
points = {
    'P1': np.array([0.1, 0.6]),
    'P2': np.array([0.15, 0.71]),
    'P3': np.array([0.08, 0.9]),
    'P4': np.array([0.16, 0.85]),
    'P5': np.array([0.2, 0.3]),
    'P6': np.array([0.25, 0.5]),
    'P7': np.array([0.24, 0.1]),
    'P8': np.array([0.3, 0.2])
}

# 2. Define Initial Centroids
m1 = points['P1'] # Cluster 1
m2 = points['P8'] # Cluster 2

print("Initial Centroid m1:", m1)
print("Initial Centroid m2:", m2)

# Initialize empty clusters
cluster1_points = []
cluster2_points = []
cluster1_labels = []
cluster2_labels = []

# Loop through each point to find the nearest centroid
for name, point in points.items():
    # Calculate Euclidean distance to m1 and m2
    # dist = sqrt((x2-x1)^2 + (y2-y1)^2)
    dist_m1 = np.sqrt(np.sum((point - m1) ** 2))
    dist_m2 = np.sqrt(np.sum((point - m2) ** 2))
    
    # Assign to nearest cluster
    if dist_m1 < dist_m2:
        cluster1_points.append(point)
        cluster1_labels.append(name)
        assigned = "Cluster 1"
    else:
        cluster2_points.append(point)
        cluster2_labels.append(name)
        assigned = "Cluster 2"
        
    print(f"{name}: Dist to m1={dist_m1:.3f}, Dist to m2={dist_m2:.3f} -> Assigned to {assigned}")

print("\n--- Cluster Membership ---")
print("Cluster 1 members:", cluster1_labels)
print("Cluster 2 members:", cluster2_labels)


# --- Question 1: Which cluster does P6 belong to? ---
p6_cluster = "Cluster 1" if 'P6' in cluster1_labels else "Cluster 2"
print(f"1] P6 belongs to: {p6_cluster}")

# --- Question 2: Population of cluster around m2 ---
# Population is simply the count of points in Cluster 2
pop_m2 = len(cluster2_points)
print(f"2] Population of cluster around m2: {pop_m2}")

# --- Question 3: Updated value of m1 and m2 ---
# Calculate the new mean (average) for each cluster
new_m1 = np.mean(cluster1_points, axis=0)
new_m2 = np.mean(cluster2_points, axis=0)

print(f"3] Updated value of m1: {new_m1}")
print(f"   Updated value of m2: {new_m2}")