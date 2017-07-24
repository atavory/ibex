.. _function_transformer:

Transforming
============


This chapter describes the :py:func:`ibex.trans` function. It allows

#. applying functions or estimators to :class:`pandas.DataFrame` objects

#. selcting a subset of columns for applications

#. naming the output columns of the results

or any combination of these.


We'll use a :class:`pandas.dataframe` ``X``, with columns ``'a'`` and ``'b'``, and (implied) index ``1, 2, 3``,

    >>> import pandas as pd
    >>> X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

and also import ``trans``:

    >>> from ibex import trans


Specifying Functions
--------------------

The (positionally first) ``func`` argument allows specifying the transformation to apply. 

This can be ``None``, meaning that the output should be the input:
    
    >>> trans().fit_transform(X)
       a  b
    0  1  3
    1  2  4

(see :ref:`function_transformer_multiple_transformations` for a use for this). It can alternatively be a function, which will be applied to the 
:attr:`pandas.DataFrame.values` of the input:

    >>> import numpy as np
    >>> trans(np.sqrt).fit_transform(X)
              a         b
    0  1.000000  1.732051
    1  1.414214  2.000000

Finally, it can be a different estimator: 

    >>> from ibex.sklearn.decomposition import PCA 
    >>> trans(PCA(n_components=2)).fit_transform(X)
              a  ...b
    0 -0.707107  ...
    1  0.707107  ...


Specifying Input Columns
------------------------

The (positionally second) ``in_cols`` argument allows specifying the columns to which to apply the function. 

If it is ``None``, then the function will be applied to all columns.

If it is a string, the function will be applied to the ``DataFrame`` consisting of the single column corresponding to this string:

    >>> trans(None, 'a').fit_transform(X)
       a
    0  1
    1  2
    >>> trans(np.sqrt, 'a').fit_transform(X)
              a
    0  1.000000
    1  1.414214


If it is a ``list`` of strings, the function will be applied to the ``DataFrame`` consisting of the columns corresponding to these strings:


    >>> trans(None, ['a']).fit_transform(X)
       a
    0  1
    1  2
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


.. _function_transformer_specifying_output_columns:

Specifying Output Columns
-------------------------

The (positionally third) ``out_cols`` argument allows specifying the names of the columns of the result. 

If it is ``None``, then the output columns will be as explained in 
:ref:`_verification_and_processing_output_dataframe_columns` 
in
:ref:`_verification_and_processing`:

    >>> trans(np.sqrt, out_cols=None).fit_transform(X)
              a         b
    0  1.000000  1.732051
    1  1.414214  2.000000

If it is a string, the function will be applied to the ``DataFrame`` consisting of the single column corresponding to this string:

    >>> trans(PCA(n_components=1), out_cols='pc').fit_transform(X)
            pc
    0 -0.707107
    1  0.707107


    >>> trans(None, out_cols=['c', 'd']).fit_transform(X)
       c  d
    0  1  3
    1  2  4

    >>> trans(np.sqrt, out_cols=['c', 'd']).fit_transform(X)
              c         d
    0  1.000000  1.732051
    1  1.414214  2.000000

    >>> trans(PCA(n_components=2), out_cols=['pc1', 'pc2']).fit_transform(X)
              pc1  pc2
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

    >>> trans(PCA(n_components=1), 'a', 'pc').fit_transform(X)
         pc
    0 -0.5
    1  0.5


.. _function_transformer_multiple_transformations:

Multiple Transformations
------------------------


