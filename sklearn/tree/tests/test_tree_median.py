"""
Testing for the tree module (sklearn.tree).
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

np.random.seed(0)

#
test_to_do = 1

if test_to_do == 1:
    dim = 1
    n_sample = 10
    x_min, x_max = -1, 1.
    tmp = [np.linspace(x_min, x_max, n_sample, dtype=np.float32)]*dim
    grid = np.meshgrid(*tmp)
    X = np.vstack(grid).reshape(dim, -1).T
    y = X**2

    median_tree = DecisionTreeRegressor(criterion="median", max_depth=1)
    median_tree.fit(X, y)
    
    thresholds = median_tree.tree_.threshold[median_tree.tree_.feature !=-2.]

    fig, ax = plt.subplots()
    if dim == 1:
        ax.plot(X, y, '.')
        for threshold in thresholds:
            ax.plot([threshold]*2, [y.min(), y.max()])
    elif dim == 2:
        ax.plot(X[:, 0], X[:, 1], '.')

    ax.axis("tight")
    fig.tight_layout()


if test_to_do == 2:
    # also load the iris dataset
    # and randomly permute it
    iris = datasets.load_iris()
    rng = np.random.RandomState(1)
    perm = rng.permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]

    X = iris.data
    y = iris.target

    median_tree = DecisionTreeRegressor(criterion="median")
    median_tree.fit(X, y, )