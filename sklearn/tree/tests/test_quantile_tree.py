import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree.tree import DecisionTreeRegressor

print "Quantile Test"

n = 1000
alpha = 0.05
x = np.linspace(0., 1., n).reshape(-1, 1)
y = x.ravel()
tree = DecisionTreeRegressor(min_samples_leaf=5).fit(x, y)
quantiles = tree.predict_quantile(x, alpha)

print quantiles