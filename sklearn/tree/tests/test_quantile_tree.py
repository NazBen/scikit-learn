import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

print "Quantile Test"

n = 10000
alpha = 0.05
x = np.linspace(0., 1., n).reshape(-1, 1)
y = x.ravel()
# tree = DecisionTreeRegressor(min_samples_leaf=3).fit(x, y)
# quantiles = tree.predict_quantile(x, alpha)

# print quantiles

forest = RandomForestRegressor(min_samples_leaf=100, n_jobs=8).fit(x, y)
quantiles_forest = forest.predict_quantile(x, alpha)

print quantiles_forest