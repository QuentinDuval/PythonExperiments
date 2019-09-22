from typing import List, Any
import numpy as np


def randomly_select(points, k):
    indexes = np.arange(len(points))
    return [points[i] for i in np.random.choice(indexes, size=k, replace=False)]


def distance_squared(p1, p2):   # Idea: show the typical change if you compute the sqrt...
    return sum((p1[i] - p2[i]) ** 2 for i in range(len(p1)))


def assign_to_nearest_centroid(points, centroids):
    groups = [[] for _ in range(len(centroids))]
    for point in points:
        min_dist = distance_squared(point, centroids[0])
        closest = 0
        for i in range(1, len(centroids)):
            dist = distance_squared(point, centroids[i])
            if dist < min_dist:
                min_dist = dist
                closest = i
        groups[closest].append(point)
    return groups


def compute_center_of_mass(points):
    # Axis 0 means first dimension, so a list of array will sum the array by index
    return np.sum(points, axis=0) / len(points)


def adjust_centroids(points, centroids):
    new_centroids = []
    groups = assign_to_nearest_centroid(points, centroids)
    for group in groups:
        if group:
            new_centroids.append(compute_center_of_mass(group))
    return new_centroids


def to_point_set(points):
    return set(tuple(point) for point in points)


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


"""
Tests
"""


def test():
    all_points = [
        np.array([1, 1, 1], dtype=np.float64),
        np.array([2, 2, 2], dtype=np.float64),
        np.array([5, 5, 5], dtype=np.float64),
        np.array([6, 6, 6], dtype=np.float64),
    ]
    all_centroids = k_means(all_points, k=2, max_iter=10)
    print(all_centroids)


def random_points(loc, scale, size):
    d = len(loc)
    points = np.zeros(shape=(size, d))
    for i in range(d):
        points[:, i] = np.random.normal(loc=loc[i], scale=scale[i], size=size)
    return points


def test_huge(size=1000):
    points1 = random_points(loc=[0., 0., 0.], scale=[1., 1., 1.], size=size // 2)
    points2 = random_points(loc=[5., 3., 6.], scale=[1., 1., 1.], size=size // 2)
    points = np.concatenate((points1, points2), axis=0)
    np.random.shuffle(points)
    centroids = k_means(points, k=2, max_iter=100)
    print(centroids)


test()
test_huge(size=10_000)
