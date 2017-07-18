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

    >>> est.transform(x[['b', 'a']])
         a    b
    0 -1.0 -1.0
    1  1.0  1.0

    >>> est.transform(x.rename(columns={'b': 'c'}))
    Traceback (most recent call last):
    ...
    KeyError: "Index(['b'], dtype='object') not in index"

