import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock, euclidean, mahalanobis,minkowski

# Load datasets
data_a = pd.read_csv("dataSet1a.csv", header=None)
data_b = pd.read_csv("dataSet1b.csv", header=None)

# Test point
point = np.array([3, 3])

# Centroids
centroid_a = data_a.mean().values
centroid_b = data_b.mean().values

print("Class a centroid:", centroid_a)
print("Class b centroid:", centroid_b)

# Manhattan distance
manhattan_a = cityblock(point, centroid_a)
manhattan_b = cityblock(point, centroid_b)

print("\nManhattan distance")
print("Class a:", manhattan_a)
print("Class b:", manhattan_b)

# Euclidean distance
euclidean_a = euclidean(point, centroid_a)
euclidean_b = euclidean(point, centroid_b)

print("\nEuclidean distance")
print("Class a:", euclidean_a)
print("Class b:", euclidean_b)

# Mahalanobis distance
cov_a = np.cov(data_a.T)
cov_b = np.cov(data_b.T)

inv_cov_a = np.linalg.inv(cov_a)
inv_cov_b = np.linalg.inv(cov_b)

mahalanobis_a = mahalanobis(point, centroid_a, inv_cov_a)
mahalanobis_b = mahalanobis(point, centroid_b, inv_cov_b)

print("\nMahalanobis distance")
print("Class a:", mahalanobis_a)
print("Class b:", mahalanobis_b)


print("\nminkowski distance values")

p_values = [0.5, 1, 1.5, 2, 100]

for p in p_values:
    dist_a = minkowski(point, centroid_a, p)
    dist_b = minkowski(point, centroid_b, p)

    closer = "Class a" if dist_a < dist_b else "Class b"

    print(f"p={p}: Class a={dist_a:.4f}, Class b={dist_b:.4f}, closer={closer}")

print("\nQuestion 2")
# Question 2
data2 = pd.read_csv("dataSet2.csv", header=None)
values = data2.iloc[:, 0]

mean_value = values.mean()
std_value = values.std()

print("\nParametric Gaussian estimate")
print("Mean:", mean_value)
print("Standard deviation:", std_value)
print("Parametric mode:", mean_value)

counts, bins, patches = plt.hist(values, bins=10, edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of dataSet2")
plt.show()

max_bin_index = np.argmax(counts)

bin_start = bins[max_bin_index]
bin_end = bins[max_bin_index + 1]
hist_mode = (bin_start + bin_end) / 2

print("\nNonparametric histogram estimate")
print("Highest bin:", bin_start, "to", bin_end)
print("Nonparametric mode estimate:", hist_mode)