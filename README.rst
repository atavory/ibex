Ibex
====


Ami Tavory, Shahar Azulay


.. image:: https://travis-ci.org/atavory/ibex.svg?branch=master  
    :target: https://travis-ci.org/atavory/ibex

.. image:: https://landscape.io/github/atavory/ibex/master/landscape.svg?style=flat
    :target: https://landscape.io/github/atavory/ibex/master

.. image:: https://coveralls.io/repos/github/atavory/ibex/badge.svg?branch=master
    :target: https://coveralls.io/github/atavory/ibex?branch=master

.. image:: http://readthedocs.org/projects/ibex/badge/?version=latest 
    :target: http://ibex.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/license-BSD--3--Clause-green.svg
    :target: https://raw.githubusercontent.com/atavory/ibex/master/LICENSE.txt


This library aims for two (somewhat independent) goals:

* providing `pandas <http://pandas.pydata.org/>`_ adapters for `estimators conforming to the sickit-learn protocol <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_, in particular those of `scikit-learn <http://scikit-learn.org/stable/>`_ itself

* allowing easier, and more succinct ways of combining estimators, features, and pipelines

(You might also want to check out the excellent `pandas-sklearn <https://pypi.python.org/pypi/sklearn-pandas>`_ which has the same aims, but takes a very different 
approach.)

The `full documentation at readthedocs <http://ibex.readthedocs.io/en/latest/?badge=latest>`_ defines these matters in detail, but the library has an extremely-small `interface <http://ibex.readthedocs.io/en/latest/overview.html>`_.


TL;DR
-----

The following short example shows the main points of the library. It is an adaptation of the scikit-learn example `Concatenating multiple feature extraction methods <http://scikit-learn.org/stable/auto_examples/feature_stacker.html>`_. In this example, we build a classifier for the `iris dataset <http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_ using a combination of `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_, `univariate feature selection <https://en.wikipedia.org/wiki/Feature_selection#Subset_selection>`_, and a `support vecor machine classifier <https://en.wikipedia.org/wiki/Support_vector_machine>`_.

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

	>>> from ibex.sklearn.svm import SVC
	>>> from ibex.sklearn.feature_selection import SelectKBest
	>>> from ibex.sklearn.decomposition import PCA

(Of course, it's possible to import steps from `sklearn` as well, and use them alongside and together with the steps of `ibex.sklearn`.):w

Finally, we construct a pipeline that, given a ``DataFrame`` of features:

* horizontally concatenates a 2-component PCA ``DataFrame``, and the best-feature ``DataFrame``, to a resulting ``DataFrame``  
* then, passes the result to a support-vector machine classifier outputting a pandas series



	>>> clf = PCA(n_components=2) + SelectKBest(k=1) | SVC(kernel="linear")



    >>> try:
    ...     from sklearn.model_selection import GridSearchCV
    ... except ImportError:
    ...     from sklearn.grid_search import GridSearchCV
    >>> param_grid = dict(
    ...     featureunion__pca__n_components=[1, 2, 3],
    ...     featureunion__selectkbest__k=[1, 2],
    ...     svc__C=[0.1, 1, 10])
    >>> GridSearchCV(clf, param_grid=param_grid).fit(iris[features], iris['class'])
    GridSearchCV(cv=None, error_score='raise',
           estimator=Pipeline(steps=[('featureunion', FeatureUnion(n_jobs=1,
    ...

`verification and processing <http://ibex.readthedocs.io/en/latest/input_verification_and_output_processing.html>`_

