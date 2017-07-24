Verification and Processing
========================================

Since ``sklearn`` is defined in terms of :class:`numpy.ndarray`, Ibex estimators perform verification and processing on their inputs and outputs. 

In this chapter we'll use a :class:`pandas.dataframe` ``X``, with columns ``'a'`` and ``'b'``, and (implied) index ``1, 2, 3``.

    >>> import pandas as pd 
    >>> X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

a scaling transformer ``trn`` which is ``fit``-ted on ``X``

    >>> from ibex.sklearn import preprocessing as pd_preprocessing
    >>> trn = pd_preprocessing.StandardScaler().fit(X)

and a linear-regression predicto ``prd`` which is ``fit``-ted on ``X`` also

    >>> from ibex.sklearn import linear_model as pd_linear_model
    >>> prd = pd_linear_model.LinearRegression().fit(X, pd.Series([3, 4]))


Input Verification
------------------

Following the call to ``fit``, we can apply further methods of ``trn`` to any ``DataFrame`` with the same column-set. For example, this is OK

    >>> X_1 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]})
    >>> trn.transform(X_1)
          a    b
    0 -1... -1...
    1  1...  1...
    2  3...  3...

but this is not

    >>> X_2 = X_1.rename(columns={'b': 'c'})
    >>> trn.transform(X_2)
    Traceback (most recent call last):
    ...
    KeyError: "...'b'...not in index"

|

Once an trnimater has been ``fit``-ed, the order of columns of further inputs will no longer matters:

    >>> trn.transform(X_1[['a', 'b']])
          a    b
    0 -1... -1...
    1  1...  1...
    2  3...  3...

    >>> trn.transform(X_1[['b', 'a']])
          a    b
    0 -1... -1...
    1  1...  1...
    2  3...  3...

The ``step`` will reorder the ``DataFrame`` to the same order of columns seen by ``fit``.


.. todo::

    For methods that take both ``X`` and ``y``, check the indexes match.


Output Processing
-----------------

Indexes
~~~~~~~

The indexes of returned ``DataFrame`` and ``Series`` objects, is that of the input:

    >>> X_1 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]}, index=[10, 20, 30])
    >>> trn.transform(X_1)
          a    b
    10 -1... -1...
    20  1...  1...
    30  3...  3...
    >>>
    >>> prd.predict(X_1)
    10    3...
    20    4...
    30    5...
    dtype: ...


``DataFrame`` Columns
~~~~~~~~~~~~~~~~~~~~~

In general, the columns of an outputted ``DataFrame`` object are those on which the estimator was ``fit``-ted:

    >>> trn.transform(X_1[['a', 'b']])
          a    b
    10 -1... -1...
    20  1...  1...
    30  3...  3...

    >>> trn.transform(X_1[['b', 'a']])
          a    b
    10 -1... -1...
    20  1...  1...
    30  3...  3...

Some outputted ``DataFrame`` objects have a number of columns that is different from that of the input. If this is the case, the resulting ``DataFrame``'s columns will all be blank strings (``''``): 

    >>> from ibex.sklearn import decomposition as pd_decomposition
    >>> pd_decomposition.PCA(n_components=1).fit(X).transform(X)
    <BLANKLINE>   
    0 -0.707107
    1  0.707107

.. seealso::
    
    
	blach
