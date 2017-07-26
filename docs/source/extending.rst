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

We think that, for each row, the mean values of ``'b'`` and ``'c'``, aggregated by ``'a'``, might make a useful feature. In ``pandas``, we could write this as follows:

    >>> df.groupby(df.a).transform(np.mean)
         b    c
    0  1.5  3.5
    1  1.0  3.0
    2  3.0  5.0
    3  1.5  3.5
    4  3.0  5.0


We now want write a transformer to do this, in order to use it for more general settings (e.g., `cross validation <http://scikit-learn.org/stable/modules/cross_validation.html>`_).


Writing A New Transformer Step
------------------------------

We can write a (slightly more general) estimator, as follows:

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
    ...         self.x_columns = X.columns # (4)
    ...         self._agg = X.groupby(df[self._group_col]).apply(self._agg_func)
    ...         return self
    ...         
    ...     def transform(self, X):
    ...         Xt = X[self.x_columns] # (5)
    ...         Xt = pd.merge(
    ...             Xt[[self._group_col]],
    ...             self._agg,
    ...             how='left')
    ...         return Xt[[c for c in Xt.columns if c != self._group_col]]


Note the following general points:

1. We subclass :class:`sklearn.base.BaseEstimator`, as this is an estimator.

2. We subclass :class:`sklearn.base.TransformerMixin`, as, in this case, this is specifically a transformer.

3. We subclass :class:`ibex.FrameMixin`, as this estimator deals with ``pandas`` entities.

4. In ``fit``, we make sure to set :py:attr:`ibex.FrameMixin.x_columns`; this will ensure that the transformer will "remember" the columns it should see in further calls.   

5. In ``transform``, we first use ``x_columns``. This will verify the columns of ``X``, and also reorder them according to the original order seen in ``fit`` (if needed). 

The rest is logic specific to this transformer. 

* In ``__init__``, the group column and aggregation function are stored. 

* In ``fit``, ``X`` is aggregated by the group column according to the aggregation function, and the result is recorded. 

* In ``transform``, ``X`` (which is not necessarily the one used in ``fit``) is left-merged with the aggregation result, and then the relevant columns of the result are returned.

| 

We can now use this as a regular step. If we fit it on ``df`` and transform it on the same ``df``, we get the result above:


    >>> GroupbyAggregator('a').fit(df).transform(df)
         b    c
    0  1.5  3.5
    1  1.0  3.0
    2  3.0  5.0
    3  1.5  3.5
    4  3.0  5.0


We can, however, now use it for fitting on one ``DataFrame``, and transforming another:

    >>> from sklearn import model_selection
    >>>
    >>> tr, te = model_selection.train_test_split(df, random_state=3)
    >>> GroupbyAggregator('a').fit(tr).transform(te)
         b    c
    0  0.0  2.0
    1  2.0  4.0






