Function Transformers
=====================

    >>> import pandas as pd
    >>> X = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})

    >>> from ibex import trans

Specifying Functions
--------------------
    
    >>> trans().fit_transform(X)
         a    b
    0  1.0  3.0
    1  2.0  4.0

    >>> import numpy as np
    >>> trans(np.sqrt).fit_transform(X)
              a         b
    0  1.000000  1.732051
    1  1.414214  2.000000

    >>> from ibex.sklearn.decomposition import PCA 
    >>> trans(PCA(n_components=2)).fit_transform(X)
              a    b
    0 -0.707107  0.0
    1  0.707107  0.0


Specifying Input Columns
------------------------

    >>> trans({'a': None}).fit_transform(X)
         a
    0  1.0
    1  2.0

    >>> trans({'a': None, 'b': np.sqrt}).fit_transform(X)
         a         b
    0  1.0  1.732051
    1  2.0  2.000000

    >>> trans({('a', 'b'): None}).fit_transform(X)
         a
    0  1.0
    1  2.0

    >>> trans({'a': np.sqrt}).fit_transform(X)
         a
    0  1.0
    1  4.0

    >>> trans({('a', 'b'): np.sqrt}).fit_transform(X)
         a
    0  1.0
    1  4.0

    >>> from ibex.sklearn.decomposition import PCA 
    >>> trans({'a': PCA(n_components=1)}).fit_transform(X)
         a
    0 -0.5
    1  0.5

    >>> from ibex.sklearn.decomposition import PCA 
    >>> trans({('a', 'b'): PCA(n_components=1)}).fit_transform(X)
         a
    0 -0.5
    1  0.5


Specifying Output Columns
-------------------------

    >>> from ibex.sklearn.decomposition import PCA 
    >>> trans({('a', 'b'): PCA(n_components=1)}).fit_transform(X)
         a
    0 -0.5
    1  0.5

Multiple Transformations
------------------------


