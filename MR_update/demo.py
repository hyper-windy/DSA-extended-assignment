from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(11)

plt.xkcd()

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 134#134
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

original_label = np.asarray([0] * N + [1] * N + [2] * N).T


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'ys', markersize=4, alpha=.8)

    plt.axis('equal')


def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]


def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis=1)


def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # take average
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))


def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)


(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])
center1 = []
center2 = []
center3 = []
for center in centers:
    center1.append(center[0])
    center2.append(center[1])
    center3.append(center[2])

center1 = np.asarray(center1)
center2 = np.asarray(center2)
center3 = np.asarray(center3)

kmeans_display(X, labels[-1])

plt.plot(center1[:, 0], center1[:, 1], "black", marker="P", ms=5)
plt.plot(center2[:, 0], center2[:, 1], "black", marker="P")
plt.plot(center3[:, 0], center3[:, 1], "black", marker="P")
plt.plot(center1[0][0], center1[0][1], "red", marker="x", ms=10)
plt.plot(center1[-1][0], center1[-1][1], "red", marker="P", ms=10)
plt.plot(center2[0][0], center2[0][1], "red", marker="x", ms=10)
plt.plot(center2[-1][0], center2[-1][1], "red", marker="P", ms=10)
plt.plot(center3[0][0], center3[0][1], "red", marker="x", ms=10)
plt.plot(center3[-1][0], center3[-1][1], "red", marker="P", ms=10)
plt.show()
