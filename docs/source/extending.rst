Extending
=========

Writing new estimators is easy. One way of doing this is by writing a `estimator conforming to the sickit-learn protocol <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_, and then wrapping it with :func:`ibex.frame` (see :ref:`adapting`). A different way is writing it directly as a :mod:`pandas` estimator. This might be the only way to go, if the logic of the estimator is ``pandas`` specific. This chapter shows how to write a new estimator from scratch.


Example Transformation
----------------------

Suppose we have a :class:`pandas.DataFrame` like this:

    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({'a': [1, 3, 2, 1, 2], 'b': range(5), 'c': range(2, 7)})
    >>> df
       a  b  c
    0  1  0  2
    1  3  1  3
    2  2  2  4
    3  1  3  5
    4  2  4  6

and notice that a ```groupby``-``transform`` transformation <https://pandas.pydata.org/pandas-docs/stable/groupby.html>`_,

    >>> pd.concat([df[['a']], df.groupby(df.a).transform(np.mean)], axis=1)
       a    b    c
    0  1  1.5  3.5
    1  3  1.0  3.0
    2  2  3.0  5.0
    3  1  1.5  3.5
    4  2  3.0  5.0

is useful in this case. We now want write a transformer to do this, in order to use it for more general settings (e.g., `cross validation <http://scikit-learn.org/stable/modules/cross_validation.html>`_).


Writing A New Transformer Step
------------------------------

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
    ...     def fit(self, X, _=None):
    ...         self.x_cols = X.columns # (4)
    ...         self._agg = X.groupby(df[self._group_col]).apply(self._agg_func)
    ...         return self
    ...         
    ...     def transform(self, X):
    ...         Xt = X[self.x_cols] # (5)
    ...         Xt = pd.merge(
    ...             Xt[[self._group_col]],
    ...             self._agg,
    ...             how='left')
    ...         return Xt


:class:`sklearn.base.BaseEstimator`, 
:class:`sklearn.base.TransformerMixin`, 
:class:`ibex.FrameMixin`, 

    >>> GroupbyAggregator('a').fit(df).transform(df)
       a    b    c
    0  1  1.5  3.5
    1  3  1.0  3.0
    2  2  3.0  5.0
    3  1  1.5  3.5
    4  2  3.0  5.0


    >>> from sklearn import model_selection
    >>>
    >>> tr, te = model_selection.train_test_split(df, random_state=3)
    >>> GroupbyAggregator('a').fit(tr).transform(te)
       a    b    c
    0  1  0.0  2.0
    1  2  2.0  4.0





