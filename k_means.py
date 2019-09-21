from typing import List, Any
import numpy as np


def randomly_select(points, k):
    centroids_indexes = np.arange(len(points))
    centroids_indexes = np.random.choice(centroids_indexes, size=k, replace=False)
    centroids = [points[i] for i in centroids_indexes]
    return centroids


def distance_between(p1, p2):
    distance = 0
    for i in range(len(p1)):
        distance = (p1[i] - p2[i]) ** 2
    return distance


def assign_to_nearest_centroid(points, centroids):
    groups = [[] for _ in range(len(centroids))]
    for p in points:
        min_dist = distance_between(p, centroids[0])
        closest = 0
        for i in range(1, len(centroids)):
            dist = distance_between(p, centroids[i])
            if dist < min_dist:
                min_dist = dist
                closest = i
        groups[closest].append(p)
    return groups


def compute_center_of_mass(points):
    total = np.sum(points, axis=0)  # Axis 0 means first dimension, so a list of array will sum the array by index
    return total / len(points)


def adjust_centroids(points, centroids):
    new_centroids = []
    groups = assign_to_nearest_centroid(points, centroids)
    for group in groups:
        if group:
            new_centroids.append(compute_center_of_mass(group))
    return new_centroids


def to_point_set(points):
    return set(tuple(point.flat) for point in points)


def k_means(points, k: int, max_iter: int) -> List[Any]:
    if k >= len(points):
        return points

    centroids = randomly_select(points, k)
    centroids_set = to_point_set(centroids)
    for _ in range(max_iter):
        new_centroids = adjust_centroids(points, centroids)
        new_centroids_set = to_point_set(new_centroids)
        if centroids_set == new_centroids_set:
            break
        centroids = new_centroids
        centroids_set = new_centroids_set
    return centroids


all_points = [
    np.array([1, 1, 1], dtype=np.float64),
    np.array([2, 2, 2], dtype=np.float64),
    np.array([5, 5, 5], dtype=np.float64),
    np.array([6, 6, 6], dtype=np.float64),
]
all_centroids = k_means(all_points, k=2, max_iter=10)
print(all_centroids)
