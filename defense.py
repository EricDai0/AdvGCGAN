from scipy.spatial import distance as set_distance
import os
import imageio
import numpy as np

def SRS(points, percentage=0.2):
    new_batch = np.zeros(points.shape)
    for j in range(points.shape[0]):
        new = None
        n = int(round(points.shape[1] * percentage))
        idx = np.arange(points.shape[1])
        np.random.shuffle(idx)
        new = np.delete(points[j], idx[:n], 0)
        s = points.shape[1]-n
        if n == 0:
            new_batch[j] = new
            continue
        n_inx = np.random.randint(low=0, high=s, size=n)
        #print(len(n_inx))
        for i in range(len(n_inx)):
            new = np.append(new, [new[n_inx[i]]], axis=0)
        new_batch[j] = new
    return new_batch

def SOR(points, alpha = 1.1, k = 2):
    new_batch = np.zeros(points.shape)
    for j in range(points.shape[0]):
        # Distances ||Xi - Xj||
        dist = set_distance.squareform(set_distance.pdist(points[j]))
        # Closest points
        closest = np.argsort(dist, axis=1)
        # Choose k neighbors
        dist_k = [sum(dist[i, closest[i,1:k]])/(k-1) for i in range(points.shape[1])]
        # Mean and standard deviation
        di = np.mean(dist_k) + alpha * np.std(dist_k)
        # Only points that have lower distance than di
        list_idx = [i for i in range(len(dist_k)) if dist_k[i] < di]
        if len(list_idx) > 0 :
        # Concatenate the new and the old indexes
            idx = np.concatenate((np.random.choice(list_idx, (points.shape[1]-np.unique(list_idx).shape[0])), list_idx))
            # New points
            new = np.array([points[j][idx[i]] for i in range(len(idx))])
        else :
            new = points[j]
        new_batch[j] = new
    return new_batch