Pipelining
==========

A pipeline is a sequential composition of a number of transformers, and a final estimator. Ibex allows pipeline compositions in both the traditional 

    >>> import numpy as np
    >>> from sklearn import datasets
    >>> import pandas as pd
    >>> 
    >>> iris = datasets.load_iris()
    >>> features, iris = iris['feature_names'], pd.DataFrame(
    ...     np.c_[iris['data'], iris['target']],
    ...     columns=iris['feature_names']+['class'])
    >>> 
    >>> iris.columns
    Index([...'sepal length (cm)', ...'sepal width (cm)', ...'petal length (cm)',
           ...'petal width (cm)', ...'class'],
          dtype='object')
