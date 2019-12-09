import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from sklearn import datasets
import matplotlib.pyplot as plt
import scikitplot as skplt

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles


# plt.figure(figsize=(8, 8))
# plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)



# plt.subplot(321)
# plt.title("One informative feature, one cluster per class", fontsize='small')
# X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=1,
#                              n_clusters_per_class=1)

def a():
    """Uniform distribution, x ∈ [−10, 1], y ∈ [17, 35]
    """
    x_samples = random.uniform(low=-10, high=1, size=300)
    y_samples = random.uniform(low=17, high=35, size=300)
    plt.scatter(x_samples, y_samples)
    plt.show()

def b():
    """(b) Gaussian with center at [5,1] and std=3.
    """
    mu = [5, 1]
    sigma = 3
    samples = random.multivariate_normal(mean=mu, )
    count, bins, ignored = plt.hist(samples, 30, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    plt.show()

def main():
    # a()
    b()

if __name__ == "__main__":
    main()