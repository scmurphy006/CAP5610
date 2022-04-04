""" Implement K-means from scratch 
        Use the following different methods for measuring distance:
            Euclidean Distance
            Cosine Similarity
            Generalized Jacard Similarity
"""    
import random as random
from math import dist
from tokenize import String

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm


class KmeansStat:
    def __init__(self, dt: String, sc: String):
        """Constructs new instance of KmeansStat

        Args:
            dt (String): The distance type
            sc (String): The stop condition
        """
        self.dt = dt
        self.sc = sc
        self.sse = None
        self.last_sse = None
        self.acc = None
        self.num_it = 0

    def header(self):
        return self.dt + "-" + self.sc + ": " + str(self.num_it)

    def status(self):
        return "ITER: " + str(self.num_it) + ", ACC: " + str(self.acc) + ", SSE: " + str(self.sse)

class Cluster:
    max_iters = 100
    def __init__(self, center: list, name: String):
        """Create a cluster

        Args:
            center (list): The initial center fo this cluster
        """
        self.center = center[1:]
        self.name = name
        self.points = []
        self.label = None
        self.center_changed = None

    def sum_distances(self, point: list, data):
        distance = 0
        for i in data:
            if i not in self.points:
                distance += dist(point, i[1:]) 
        return distance

    
    def re_calculate_center(self, distance_type, data):
        # Also determine the majority label
        if len(self.points) > 0:
            labels = list(list(zip(*self.points))[0])
            labels = list(np.concatenate(labels).flat)
            self.label = max(set(labels), key = labels.count)
        
            # Update center
            old_center = self.center
            if distance_type == "JACCARD":
                # Find medoid
                current_min = None
                medoid = None
                for p in self.points:
                    without_label = p[1:]
                    distance = self.sum_distances(without_label, data)
                    if current_min is None or distance < current_min:
                        medoid = without_label
                        current_min = distance
                self.center = medoid
            else:
                self.center = [np.mean(k) for k in zip(*self.points)][1:]

            # If the euclidean distance between the old center and the new center is > 0, return true
            euc = dist(old_center, self.center)
            print(self.name + "(" + str(len(self.points)) +")"+ ": " + str(euc))
            if dist(old_center, self.center) > 0:
                self.center_changed = True
            else:
                self.center_changed = False
        else:
            print(self.name + "(0): N/A")
            self.center_changed = False

    def get_num_correct(self):
        num_correct = 0
        for p in self.points:
            if p[0][0] == self.label:
                num_correct += 1
        return num_correct

        
# Hardcoded k to the values of y
K = 10

N = 10000

DATA_LOCAITON = "HW4\data.csv"

LABEL_LOCAITON = "HW4\label.csv"

DISTANCE_TYPES = ["EUCLIDEAN", "JACCARD", "COSINE"]

STOP_CONDITIONS = ["CENTROID", "SSE", "MAX_ITERATIONS", "ANY"]

STATS = []

CLUSTERS = []

def jaccard(a: list, b: list):
    s1 = set(a)
    s2 = set(b)
    return 1- float(len(s1.intersection(s2)) / len(s1.union(s2)))

def compute_distance(distance_type: String, a: list, b: list) -> float:
    if distance_type == "EUCLIDEAN":
        return dist(a, b)
    elif distance_type == "JACCARD":
        return jaccard(a, b)
    elif distance_type == "COSINE":
        return 1 - dot(a, b)/(norm(a)*norm(b))
    return 0

def centroids_changed():
    changed = False
    for c in CLUSTERS:
        if c.center_changed:
            changed = True
    return changed

def should_stop(stop_cond: String, stats: KmeansStat) -> bool:
    centroids = not centroids_changed()
    sse = not (stats.last_sse is None or stats.sse < stats.last_sse)
    max_iters = not stats.num_it < 10
    if stop_cond == "CENTROID":
        return centroids
    elif stop_cond == "SSE":
        return sse
    elif stop_cond == "MAX_ITERATIONS":
        return max_iters
    else:
        return centroids or sse or max_iters

def assign_to_cluster(distance_type: String, point: list) -> Cluster:
    """Assign the passed in point to the closest cluster

    Args:
        distance_type (String): The distance type 
        point (list): The current point
    """
    without_label = point[1:]
    current_min = None
    closest_cluster = None
    for c in CLUSTERS:
        distance = compute_distance(distance_type, without_label, c.center)
        if current_min is None or distance < current_min:
            closest_cluster = c
            current_min = distance

    # Assign point to the closest cluster
    closest_cluster.points.append(point)

def calc_sse():
    sse = 0
    for c in CLUSTERS:
        for p in c.points:
            sse += dist(p[1:], c.center)**2
    return sse

def rinse_and_repeate(stats: KmeansStat, distance_type: String, stop_cond: String, data: list) -> KmeansStat:
    """Repeate
          Assign each point to its closest centroid
          Compute the new centroid (mean) of each cluster
          Stop when the defined stop condition is met

    Args:
        distance_type (String): The distance type
        stop_cond (String): The stop condition
        df (list): data
    Returns:
        KmeansStat: The stats for this run on k-means
    """
    # Increment the iterations
    stats.num_it += 1 
    print("\r" + stats.header(), end="",
                        flush=True,)
    print("\n" + str(stats.sse))

    # Clear the points in each cluster
    for c in CLUSTERS:
        c.points = []

    # Assign each point to its closest centroid
    for record in data:
        assign_to_cluster(distance_type, record)

    # Recalculate the center of each cluster
    print('\n')
    for c in CLUSTERS:
        c.re_calculate_center(distance_type, data)

    # Update SSE
    stats.last_sse = stats.sse
    stats.sse = calc_sse()

    # Check stop condition
    if should_stop(stop_cond, stats):
        return stats
    return rinse_and_repeate(stats, distance_type, stop_cond, data)

def kmeans(distance_type: String, stop_cond: String, data: list):
    """Apply the k-means algorithm to the passed in dataframe.
       Use the defined distance type and stop condition.
       Record SSE, predictive accuracy, and number of iterations
       Kmeans-algorithm:
       Randomly initialize k centroids
       Repeate
          Assign each point to its closest centroid
          Compute the new centroid (mean) of each cluster
          Stop when the defined stop condition is met
    Args:
        distance_type (String): The distance type
        stop_cond (String): The stop condition
        df (list): data
    """
    print("\n------------------------------")
    # Randomly initialize k centroids
    centroids = random.sample(range(0, N), 10)

    # Create clusters from our centroids
    for i in centroids:
        CLUSTERS.append(Cluster(data[i], str(i)))
    # Repeate the process of k-means until the stop condition has been met 
    stats = rinse_and_repeate(KmeansStat(distance_type, stop_cond), distance_type, stop_cond, data)

    # Determine accuracy
    num_correct = 0
    for c in CLUSTERS:
        num_correct += c.get_num_correct()

    stats.acc = num_correct / 10000
    print("\n" + stats.status())

def main():
    """For each distance type for each stop condition
       run K-means and measure the accuracy
    """
    labels = pd.read_csv(LABEL_LOCAITON, header=None)
    df = pd.read_csv(DATA_LOCAITON, header=None)
    df.insert(0, "Label", labels.values.tolist(), True)

    for dt in DISTANCE_TYPES:
        for sc in STOP_CONDITIONS:
            CLUSTERS.clear()
            data = df.values.tolist()[:N]
            # Fit_Predict with all our data
            kmeans(dt, sc, data)

if __name__ == "__main__":
    main()
