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

Note that the name of the second step is ``'selectkbest'``:

    >>> trn.transformer_list
    [('pca', Adapter[PCA](...
      ...), ('selectkbest', Adapter[SelectKBest](...)]


This is `because the name of the class (in lowercase) <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html>`_ is ``'selectkbest'``:

    >>> PDSelectKBest.__name__.lower()
    'selectkbest'

In fact, this is exactly the behavior of :func:`sklearn.pipeline.make_pipeline`. The ``make_pipeline`` function, however, does not allow using same-class objects, as the names would be duplicated. Ibex allows this by detecting this, and numbering same-class steps:

    >>> from ibex import trans
    >>>
    >>> (trans(np.sin) | trans(np.cos)). steps
    [('functiontransformer_0', FunctionTransformer(...
              ...)), ('functiontransformer_1', FunctionTransformer(...
              ...))]
    >>>
    >>> (trans(np.sin) | trans(np.cos) | trans(np.tan)). steps
    [('functiontransformer_0', FunctionTransformer(...
              ...)), ('functiontransformer_1', FunctionTransformer(...
              ...)), ('functiontransformer_2', FunctionTransformer(...
              ...))]

This alternative, therefore, is more succinct, but allows less control over the steps' names.

