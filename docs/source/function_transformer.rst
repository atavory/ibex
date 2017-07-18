Function Transformers
=====================

    >>> import pandas as pd
    >>> x = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})

    >>> from ibex import trans

Specifying Functions
--------------------
    
    >>> trans().fit_transform(x)
         a    b
    0  1.0  3.0
    1  2.0  4.0

    >>> trans(lambda df: df ** 2).fit_transform(x)
         a     b
    0  1.0   9.0
    1  4.0  16.0

    >>> from ibex.sklearn.decomposition import PCA 
    >>> trans(PCA(n_components=2)).fit_transform(x)
              a    b
    0 -0.707107  0.0
    1  0.707107  0.0


Specifying Column Subsets
-------------------------

    >>> trans(columns=['a']).fit_transform(x)
         a
    0  1.0
    1  2.0

    >>> trans(lambda df: df ** 2, columns=['a']).fit_transform(x)
         a
    0  1.0
    1  4.0

    >>> from ibex.sklearn.decomposition import PCA 
    >>> trans(PCA(n_components=1), columns=['a']).fit_transform(x)
         a
    0 -0.5
    1  0.5




