Ibex
====


Ami Tavory, Shahar Azulay, Tali Raveh-Sadka

.. image:: https://travis-ci.org/atavory/ibex.svg?branch=master  
    :target: https://travis-ci.org/atavory/ibex

.. image:: https://landscape.io/github/atavory/ibex/master/landscape.svg?style=flat
    :target: https://landscape.io/github/atavory/ibex/master

.. image:: https://img.shields.io/codecov/c/github/atavory/ibex/master.svg
    :target: https://codecov.io/gh/atavory/ibex/

.. image:: http://readthedocs.org/projects/ibex/badge/?version=latest 
    :target: https://atavory.github.io/ibex/

.. image:: https://img.shields.io/badge/license-BSD--3--Clause-brightgreen.svg
    :target: https://raw.githubusercontent.com/atavory/ibex/master/LICENSE.txt

.. image:: https://badge.fury.io/py/ibex.svg
    :target: https://badge.fury.io/py/ibex


This library aims for two (somewhat independent) goals:

* providing `pandas <http://pandas.pydata.org/>`_ adapters for `estimators conforming to the scikit-learn protocol <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_, in particular those of `scikit-learn <http://scikit-learn.org/stable/>`_ itself

* providing easier, and more succinct ways of combining estimators, features, and pipelines

(You might also want to check out the excellent `pandas-sklearn <https://pypi.python.org/pypi/sklearn-pandas>`_ which has the same aims, but takes a very different 
approach.)

The full documentation at |github_pages_ibex|_ defines these matters in detail, but the library has an extremely-small `interface <https://atavory.github.io/ibex/overview.html>`_.


.. |github_pages_ibex| image:: http://ibex.readthedocs.io/en/latest/_static/github-pages.png
.. _github_pages_ibex: https://atavory.github.io/ibex/


TL;DR
-----

The following short example shows the main points of the library. It is an adaptation of the scikit-learn example `Concatenating multiple feature extraction methods <http://scikit-learn.org/stable/auto_examples/plot_feature_stacker.html#sphx-glr-auto-examples-plot-feature-stacker-py>`_. In this example, we build a classifier for the `iris dataset <http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_ using a combination of `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_, `univariate feature selection <https://en.wikipedia.org/wiki/Feature_selection#Subset_selection>`_, and a `support vecor machine classifier <https://en.wikipedia.org/wiki/Support_vector_machine>`_.

We first load the Iris dataset into a pandas ``DataFrame``.

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

Now, we import the relevant steps. Note that, in this example, we import them from `ibex.sklearn` rather than `sklearn`.

	>>> from ibex.sklearn.svm import SVC as PdSVC
	>>> from ibex.sklearn.feature_selection import SelectKBest as PdSelectKBest
	>>> from ibex.sklearn.decomposition import PCA as PdPCA

(Of course, it's possible to import steps from `sklearn` as well, and use them alongside and together with the steps of `ibex.sklearn`.)

Finally, we construct a pipeline that, given a ``DataFrame`` of features:

* horizontally concatenates a 2-component PCA ``DataFrame``, and the best-feature ``DataFrame``, to a resulting ``DataFrame``  
* then, passes the result to a support-vector machine classifier outputting a pandas series:


	>>> clf = PdPCA(n_components=2) + PdSelectKBest(k=1) | PdSVC(kernel="linear")

``clf`` is now a ``pandas``-ware classifier, but otherwise can be used pretty much like all ``sklearn`` estimator. For example,  

    >>> param_grid = dict(
    ...     featureunion__pca__n_components=[1, 2, 3],
    ...     featureunion__selectkbest__k=[1, 2],
    ...     svc__C=[0.1, 1, 10])
    >>> try:
    ...     from ibex.sklearn.model_selection import GridSearchCV as PdGridSearchCV
    ... except: # Accomodate older versions of sklearn
    ...     from ibex.sklearn.grid_search import GridSearchCV as PdGridSearchCV
    >>> PdGridSearchCV(clf, param_grid=param_grid).fit(iris[features], iris['class']) # doctest: +SKIP 
    ...

So what does this add to the original version?

#. The estimators perform `verification and processing <https://atavory.github.io/ibex/input_verification_and_output_processing.html>`_ on the inputs and outputs. They verify column names following calls to ``fit``, and index results according to those of the inputs. This helps catch bugs.

#. It allows `writing Pandas-munging estimators <https://atavory.github.io/ibex/extending.html>`_ (see also `Multiple-Row Features In The Movielens Dataset <movielens_simple_row_aggregating_features.ipynb>`_).

#. Using ``DataFrame`` metadata, it allows writing more complex meta-learning algorithms, such as stacking and nested labeled and stratified cross validation.

#. The pipeline syntax is succinct and clear (see `Motivation For Shorter Combinations <https://atavory.github.io/ibex/pipeline_motivation.html>`_).
