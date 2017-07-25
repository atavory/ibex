.. _feature_union:

Uniting Features
================


A feature-union `horizontally concatenates <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html>`_ the :class:`pandas.DataFrame` results of multiple transformer objects. 

This estimator applies a list of transformer objects in parallel to the input data, then concatenates the results. This is useful to combine several feature extraction mechanisms into a single transformer.

In this chapter we'll use the following Iris dataset:

    >>> import numpy as np
    >>> from sklearn import datasets
    >>> import pandas as pd
    >>> 
    >>> iris = datasets.load_iris()
    >>> features, iris = iris['feature_names'], pd.DataFrame(
    ...     np.c_[iris['data'], iris['target']],
    ...     columns=iris['feature_names']+['class'])
    >>> 
    >>> iris.columns
    Index([...'sepal length (cm)', ...'sepal width (cm)', ...'petal length (cm)',
           ...'petal width (cm)', ...'class'],
          dtype='object')

We'll also use PCA and univariate feature selection:

	>>> from ibex.sklearn.decomposition import PCA as PDPCA
	>>> from ibex.sklearn.feature_selection import SelectKBest as PDSelectKBest


``sklearn`` Alternative
-----------------------

Using :class:`sklearn.pipeline.FeatureUnion`, we can create a feature-union of steps:

    >>> from sklearn.pipeline import FeatureUnion
    >>> 
    >>> trn = FeatureUnion([('pca', PDPCA(n_components=2)), ('best', PDSelectKBest(k=1))])

Note how the step names can be exactly specified. The name of the second step is ``'best'``, even though that is unrelated to the name of the class.

    >>> trn.transformer_list
    [('pca', Adapter[PCA](...
      ...), ('best', Adapter[SelectKBest](...)]

.. tip::

    Steps' names are important, as they are `used by <http://scikit-learn.org/stable/modules/pipeline.html>`_ 
    :meth:`ibex.sklearn.pipeline.FeatureUnion.set_params` and :meth:`ibex.sklearn.pipeline.FeatureUnion.get_params`.


Pipeline-Syntax Alternative
---------------------------

Using the pipeline syntax, we can use ``+`` to create a pipeline:

	>>> trn = PDPCA(n_components=2) + PDSelectKBest(k=1)

The output using this, however, discards the meaning of the columns:

	>>> trn = PDPCA(n_components=2) + PDSelectKBest(k=1)
    >>> trn.fit_transform(iris[features], iris['class'])
    <BLANKLINE>
    0   -2.684207  0.326607  1.4
    1   -2.715391 -0.169557  1.4
    2   -2.889820 -0.137346  1.3
    3   -2.746437 -0.311124  1.5
    4   -2.728593  0.333925  1.4
	...

A better way would be to combine this with :func:`ibex.trans`:

	>>> from ibex import trans
	>>> 
	>>> trn = trans(PDPCA(n_components=2), out_cols=['pc1', 'pc2']) + trans(PDSelectKBest(k=1), out_cols='best', pass_y=True)
    >>> trn.fit_transform(iris[features], iris['class'])
              pc1       pc2  best
    0   -2.684207  0.326607   1.4
    1   -2.715391 -0.169557   1.4
    2   -2.889820 -0.137346   1.3
    3   -2.746437 -0.311124   1.5
    4   -2.728593  0.333925   1.4
	...
 

Note the names of the transformers:

    >>> trn.transformer_list
    [('functiontransformer_0', FunctionTransformer(func=Adapter[PCA](...
      ...
              ...
              ...)), ('functiontransformer_1', FunctionTransformer(func=Adapter[SelectKBest](...
              ...))]

This is similar to the discussion of :ref:`pipeline_pipeline_syntax_alternative` in :ref:`pipeline`.

.. note::

    Just as with :class:`sklearn.pipeline.Pipeline` vs. ``|``, also :class:`sklearn.pipeline.FeatureUnion` gives greater control over steps name
    relative to ``+``. Note, however that ``FeatureUnion`` provides control over further aspects, e.g., the ability to run steps in parallel.
