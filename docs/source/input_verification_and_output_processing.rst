Input Verification And Output Processing
========================================

    >>> import pandas as pd 
    >>> X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

    >>> from ibex.sklearn import preprocessing
    >>> est = preprocessing.StandardScaler().fit(X)

    >>> X_1 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]}, index=[10, 20, 30])
    >>> est.transform(X_1)
          a    b
    10 -1.0 -1.0
    20  1.0  1.0
    30  3.0  3.0

    >>> est.transform(X_1[['b', 'a']])
          a    b
    10 -1.0 -1.0
    20  1.0  1.0
    30  3.0  3.0

    >>> X_2 = X_1.rename(columns={'b': 'c'})
    >>> est.transform(X_2)
    Traceback (most recent call last):
    ...
    KeyError: "...'b'...not in index"

    >>> from ibex.sklearn import decomposition
    >>> decomposition.PCA(n_components=1).fit(X).transform(X)
    <BLANKLINE>   
    0 -0.707107
    1  0.707107


