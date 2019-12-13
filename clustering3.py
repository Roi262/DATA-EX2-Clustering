import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.pyplot as plt
import scikitplot as skplt
#from pandas import DataFrame
# from sklearn.datasets import make_classification
# from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles
from sklearn.datasets import load_sample_image, load_sample_images
from sklearn.datasets import make_moons


plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)


# a
plt.subplot(331)
plt.title("a - Uniform distribution, x ∈ [−10, 1], y ∈ [17, 35]", fontsize='small')
x_samples = random.uniform(low=-10, high=1, size=300)
y_samples = random.uniform(low=17, high=35, size=300)
plt.scatter(x_samples, y_samples)

# b
plt.subplot(332)
plt.title("b - Gaussian with center at [5,1] and std=3", fontsize='small')
X1, Y1 = make_gaussian_quantiles(mean=[5,1], cov=3, n_samples=300, n_classes=1)  #(n_features=1, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

# c: i=1
plt.subplot(333)
plt.title("c, i=1 - Gaussian with center at [1,-1] and std=.5", fontsize='small')
X1, Y1 = make_gaussian_quantiles(mean=[1,-1], cov=.5, n_samples=300, n_classes=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

# c: i=2
plt.subplot(334)
plt.title("c, i=2 - Gaussian with center at [2,-2] and std=1", fontsize='small')
X1, Y1 = make_gaussian_quantiles(mean=[2,-2], cov=1, n_samples=300, n_classes=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

# c: i=5
plt.subplot(335)
plt.title("c, i=5 - Gaussian with center at [5,-5] and std=2.5", fontsize='small')
X1, Y1 = make_gaussian_quantiles(mean=[5,-5], cov=2.5, n_samples=300, n_classes=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

# d
plt.subplot(336)
plt.title("A circle inside a ring", fontsize='small')
X1, Y1 = make_circles(n_samples=300, factor=.5, noise=.05)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

# (e) The first letter of your first names (so the data should 
# look like a noisy version of “NDH”, for your own letters; 
# if they are the same letter, use last names. :) ).
plt.subplot(337)
plt.title("e - data in form of letters TRM", fontsize='small')

# draw the T

mu_T_1 = [0, 0]
sigma_T_1 = np.array([[0, 5],
                  [.05, 0]])
x, y = np.random.multivariate_normal(mu_T_1, sigma_T_1, 20).T
plt.scatter(x, y)

mu_T_2 = [0, 5]
sigma_T_2 = np.array([[0, .05],
                      [3, 0]])
x2, y2 = np.random.multivariate_normal(mu_T_2, sigma_T_2, 20).T
plt.scatter(x2, y2)

# draw the R

mu_R_1 = [4, 0]
sigma_R_1 = np.array([[0, 5],
                  [.05, 0]])
x3, y3 = np.random.multivariate_normal(mu_R_1, sigma_R_1, 20).T
plt.scatter(x3, y3)

mu_R_2 = [5, 5]
sigma_R_2 = np.array([[0, .05],
                      [2, 0]])
x4, y4 = np.random.multivariate_normal(mu_R_2, sigma_R_2, 20).T
plt.scatter(x4, y4)


"""
plt.subplot(337)
X, y = make_moons(n_samples=300, noise=0.1)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
"""

plt.show()
