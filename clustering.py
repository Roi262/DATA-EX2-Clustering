from sklearn import datasets
import matplotlib.pyplot as plt
import scikitplot as skplt

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles


plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)




# # plt.subplot(321)
# # plt.title("One informative feature, one cluster per class", fontsize='small')
# # X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=1,
# #                              n_clusters_per_class=1)
# # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
# #             s=25, edgecolor='k')

# # plt.subplot(322)
# # plt.title("Two informative features, one cluster per class", fontsize='small')
# # X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
# #                              n_clusters_per_class=1)
# # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
# #             s=25, edgecolor='k')


# # plt.subplot(322)
# # plt.title("Circles", fontsize='small')
# # X1, Y1 = make_circles(n_samples=300)
# # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
# #             s=25, edgecolor='k')

# # plt.subplot(323)
# # plt.title("Circles", fontsize='small')
# # X1, Y1 = make_circles(n_samples=300)
# # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
# #             s=25, edgecolor='k')



# # plt.subplot(323)
# # plt.title("Two informative features, two clusters per class",
# #           fontsize='small')
# # X2, Y2 = make_classification(n_features=2, n_redundant=0, n_informative=2)
# # plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2,
# #             s=25, edgecolor='k')

# # plt.subplot(324)
# # plt.title("Multi-class, two informative features, one cluster",
# #           fontsize='small')
# # X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
# #                              n_clusters_per_class=1, n_classes=3)
# # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
# #             s=25, edgecolor='k')

# # plt.subplot(325)
# # plt.title("Three blobs", fontsize='small')
# # X1, Y1 = make_blobs(n_features=2, centers=3)
# # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
# #             s=25, edgecolor='k')


# ##### b, c (Gaussians)

# # plt.subplot(326)
# # plt.title("Gaussian with center at [5,1] and std=3", fontsize='small')
# # X1, Y1 = make_gaussian_quantiles(n_features=1, n_classes=3)
# # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
# #             s=25, edgecolor='k')

# plt.show()


n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)