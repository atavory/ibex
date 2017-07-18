Input Verification And Output Processing
----------------------------------------

    >>> import pandas as pd 
    >>> x = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})

    >>> from ibex.sklearn import preprocessing
    >>> est = preprocessing.StandardScaler().fit(x)

    >>> est.transform(x)

    >>> est.transform(x[['b', 'a']])
