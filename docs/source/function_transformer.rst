Function Transformers
=====================

    >>> import pandas as pd
    >>> X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

    >>> from ibex import trans


Specifying Functions
--------------------
    
    >>> trans().fit_transform(X)
       a  b
    0  1  3
    1  2  4

    >>> import numpy as np
    >>> trans(np.sqrt).fit_transform(X)
              a         b
    0  1.000000  1.732051
    1  1.414214  2.000000

    >>> from ibex.sklearn.decomposition import PCA 
    >>> trans(PCA(n_components=2)).fit_transform(X)
              a  ...b
    0 -0.707107  ...
    1  0.707107  ...


Specifying Input Columns
------------------------

    >>> trans(None, 'a').fit_transform(X)
       a
    0  1
    1  2

    >>> trans(None, ['a']).fit_transform(X)
       a
    0  1
    1  2

    >>> trans(np.sqrt, 'a').fit_transform(X)
              a
    0  1.000000
    1  1.414214

    >>> trans(np.sqrt, ['a']).fit_transform(X)
              a
    0  1.000000
    1  1.414214

    >>> trans(PCA(n_components=1), 'a').fit_transform(X)
         a
    0 -0.5
    1  0.5

    >>> trans(PCA(n_components=1), ['a']).fit_transform(X)
         a
    0 -0.5
    1  0.5


Specifying Output Columns
------------------------

    >>> trans(None, out_cols=['c', 'd']).fit_transform(X)
       c  d
    0  1  3
    1  2  4

    >>> trans(np.sqrt, out_cols=['c', 'd']).fit_transform(X)
              c         d
    0  1.000000  1.732051
    1  1.414214  2.000000

    >>> trans(PCA(n_components=2), out_cols=['c', 'd']).fit_transform(X)
              c  ...d
    0 -0.707107  ...
    1  0.707107  ...


Specifying Output and Input Columns
-----------------------------------

    >>> trans(None, 'a', 'c').fit_transform(X)
       c
    0  1
    1  2

    >>> trans(None, ['a'], ['c']).fit_transform(X)
       c
    0  1
    1  2

    >>> trans(np.sqrt, ['a', 'b'], ['c', 'd']).fit_transform(X)
              c         d
    0  1.000000  1.732051
    1  1.414214  2.000000

    >>> trans(PCA(n_components=1), 'a', 'c').fit_transform(X)
         c
    0 -0.5
    1  0.5


Multiple Transformations
------------------------


