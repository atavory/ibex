Input Verification And Output Processing
========================================

    >>> import pandas as pd 
    >>> x = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})

    >>> from ibex.sklearn import preprocessing
    >>> est = preprocessing.StandardScaler().fit(x)

    >>> x_1 = pd.DataFrame({'a': [1., 2., 3.], 'b': [3., 4., 5.]}, index=[10, 20, 30])
    >>> est.transform(x_1)
          a    b
    10 -1.0 -1.0
    20  1.0  1.0
    30  3.0  3.0

    >>> est.transform(x_1[['b', 'a']])
          a    b
    10 -1.0 -1.0
    20  1.0  1.0
    30  3.0  3.0

    >>> x_2 = x_1.rename(columns={'b': 'c'})
    >>> est.transform(x_2)
    Traceback (most recent call last):
    ...
    KeyError: "Index([...'b'], dtype='object') not in index"

    >>> from ibex.sklearn import decomposition
    >>> decomposition.PCA(n_components=1).fit(x).transform(x)
    <BLANKLINE>   
    0 -0.707107
    1  0.707107


