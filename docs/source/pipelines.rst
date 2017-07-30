.. _pipeline:

Pipelining
==========

A pipeline is a sequential composition of a number of transformers, and a final estimator. Ibex allows pipeline compositions in both the original ``sklearn``  explicit way, as well as a more succinct pipeline-syntax version.

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

We'll also use SVC and PCA:

	>>> from ibex.sklearn.svm import SVC as PDSVC
	>>> from ibex.sklearn.decomposition import PCA as PDPCA


``sklearn`` Alternative
-----------------------

Using :class:`sklearn.pipeline.Pipeline`, we can create a pipeline of steps:

    >>> from sklearn.pipeline import Pipeline
    >>> 
    >>> clf = Pipeline([('pca', PDPCA(n_components=2)), ('svm', PDSVC(kernel="linear"))])

Note how the step names can be exactly specified. The name of the second step is ``'svm'``, even though that is unrelated to the name of the class.

    >>> clf.steps
    [('pca', Adapter[PCA](...
      ...)), ('svm', Adapter[SVC](...
	  ...
	  ...
      ...))]

.. tip::

    Steps' names are important, as they are `used by <http://scikit-learn.org/stable/modules/pipeline.html>`_ 
    :meth:`sklearn.pipeline.Pipeline.set_params` and :meth:`sklearn.pipeline.Pipeline.get_params`.



.. pipeline_pipeline_syntax_alternative:

Pipeline-Syntax Alternative
---------------------------

Using the pipeline syntax, we can use ``|`` to create a pipeline:

	>>> clf = PDPCA(n_components=2) | PDSVC(kernel="linear")

Note that the name of the second step is ``'svc'``:

    >>> clf.steps
    [('pca', Adapter[PCA](...
      ...)), ('svc', Adapter[SVC](...
	  ...
	  ...
      ...))]

This is `because the name of the class (in lowercase) <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html>`_ is ``'svc'``:

    >>> PDSVC.__name__.lower()
    'svc'

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

