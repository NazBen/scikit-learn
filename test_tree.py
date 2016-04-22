from sklearn.externals.six import StringIO  
import pydot
from IPython.display import Image  
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz

def sum_func(X, a=2):
    X = np.asarray(X)
    return a*X

n = 6
X = np.random.uniform(0, 10, (n)).reshape(n, 1)
X = np.linspace(0, 10, (n)).reshape(n, 1)
y = sum_func(X)

tree_median = DecisionTreeRegressor("median").fit(X, y)
print tree_median.predict(X)
dot_data = StringIO()
export_graphviz(tree_median, out_file=dot_data,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())