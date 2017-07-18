Input Verification And Output Processing
========================================

    >>> import pandas as pd 
    >>> x = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})

    >>> from ibex.sklearn import preprocessing
    >>> est = preprocessing.StandardScaler().fit(x)

    >>> est.transform(x)
         a    b
    0 -1.0 -1.0
    1  1.0  1.0

    >>> x_1 = pd.DataFrame({'a': [1., 2., 3.], 'b': [3., 4., 5.]})
    >>> est.transform(x_1[['b', 'a']])
         a    b
    0 -1.0 -1.0
    1  1.0  1.0
    2  3.0  3.0

    >>> x_2 = x_1.rename(columns={'b': 'c'})
    >>> est.transform(x_2)
    Traceback (most recent call last):
    ...
    KeyError: "Index(['b'], dtype='object') not in index"

