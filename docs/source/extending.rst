Extending
=========


    >>> import numpy as np
    >>> import pandas as pd

    >>> df = pd.DataFrame({'a': [1, 2, 1, 1, 2, 2], 'b': range(6), 'c': range(2, 8)})
    >>> df
       a  b  c
    0  1  0  2
    1  2  1  3
    2  1  2  4
    3  1  3  5
    4  2  4  6
    5  2  5  7

    >>> df[['a', 'b', 'c']].groupby('a').transform(np.mean)
              b         c
    0  1.666667  3.666667
    1  3.333333  5.333333
    2  1.666667  3.666667
    3  1.666667  3.666667
    4  3.333333  5.333333
    5  3.333333  5.333333

    >>> from sklearn import base                                                
    >>> import ibex                                                             

    >>> class GroupbyAggregator(                                                
    ...            base.BaseEstimator, # (1)
    ...            base.TransformerMixin, # (2)
    ...            ibex.FrameMixin): # (3)  
    ...        
    ...     def __init__(self, group_col, agg_func=np.mean):
    ...         self._group_col, self._agg_func = group_col, agg_func
    ...
    ...     def fit(self, X, *args):
    ...         self.x_cols = X.columns # (4)
    ...         return self
    ...         
    ...     def transform(self, X):
    ...         return X[self.x_cols].groupby(self._group_col).transform(self._agg_func) # (5)

:class:`sklearn.base.BaseEstimator`, 
:class:`sklearn.base.TransformerMixin`, 
:class:`ibex.FrameMixin`, 

    >>> GroupbyAggregator('a').fit(df).transform(df)
              b         c
    0  1.666667  3.666667
    1  3.333333  5.333333
    2  1.666667  3.666667
    3  1.666667  3.666667
    4  3.333333  5.333333
    5  3.333333  5.333333

