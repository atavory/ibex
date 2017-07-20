Ibex
====

Ami Tavory and Shahar Azulay

.. image:: https://travis-ci.org/atavory/ibex.svg?branch=master  
    :target: https://travis-ci.org/atavory/ibex

.. image:: https://landscape.io/github/atavory/ibex/master/landscape.svg?style=flat
    :target: https://landscape.io/github/atavory/ibex/master

.. image:: https://coveralls.io/repos/github/atavory/ibex/badge.svg?branch=master
    :target: https://coveralls.io/github/atavory/ibex?branch=master

.. image:: http://readthedocs.org/projects/ibex/badge/?version=latest 
    :target: http://ibex.readthedocs.io/en/latest/?badge=latest


`scikit-learn <http://scikit-learn.org/stable/>`_

`pandas <http://pandas.pydata.org/>`_

`documentation at readthedocs <http://ibex.readthedocs.io/en/latest/?badge=latest>`_


We first load the `iris dataset <http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_:

    >>> import numpy as np
    >>> from sklearn import datasets
    >>> import pandas as pd
    >>> 
    >>> iris = datasets.load_iris()
    >>> iris = pd.DataFrame(
    ...     np.c_[iris['data'], iris['target']],
    ...     columns=iris['feature_names']+['class'])
    >>> 
    >>> iris.columns
    Index([...'sepal length (cm)', ...'sepal width (cm)', ...'petal length (cm)',
           ...'petal width (cm)', ...'class'],
          dtype='object')

	>>> from ibex.sklearn.svm import SVC
	>>> from ibex.sklearn.decomposition import PCA
	>>> from ibex.sklearn.feature_selection import SelectKBest

	>>> clf = (PCA(n_components=2) + SelectKBest(k=1)) | SVC(kernel="linear")



Contents
========

.. toctree::
    :maxdepth: 2
    
    input_verification_and_output_processing
    function_transformer
    feature_union
    examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

