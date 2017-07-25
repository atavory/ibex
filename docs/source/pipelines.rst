Pipelining
==========

A pipeline is a sequential composition of a number of transformers, and a final estimator. Ibex allows pipeline compositions in both the original ``sklearn``  explicit way, as well as a more succinct pipeline-syntax version.

In this chapter we'll use the following:

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

	>>> from ibex.sklearn.svm import SVC as PDSVC
	>>> from ibex.sklearn.decomposition import PCA as PDPCA


``sklearn`` Alternative
-----------------------

    >>> from sklearn.pipeline import Pipeline

    >>> clf = Pipeline([('pca', PDPCA(n_components=2)), ('svm', PDSVC(kernel="linear"))])

    >>> clf.steps
    [('pca', Adapter[PCA](...
      ...)), ('svm', Adapter[SVC](...
	  ...
	  ...
      ...))]


Pipeline-Syntax Alternative
---------------------------

	>>> clf = PDPCA(n_components=2) | PDSVC(kernel="linear")

    >>> clf.steps
    [('pca', Adapter[PCA](...
      ...)), ('svc', Adapter[SVC](...
	  ...
	  ...
      ...))]


:func:`sklearn.pipeline.make_pipeline`


