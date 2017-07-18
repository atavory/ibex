`ibex` - `pandas` adapters for `scikit-learn`
===================================================

Ami Tavory, Shahar Azulay, and Tali Raveh-Sadka

.. image:: https://travis-ci.org/atavory/ibex.svg?branch=master  
    :target: https://travis-ci.org/atavory/ibex

.. image:: https://landscape.io/github/atavory/ibex/master/landscape.svg?style=flat
   :target: https://landscape.io/github/atavory/ibex/master

.. image:: https://coveralls.io/repos/github/atavory/ibex/badge.svg?branch=master
	:target: https://coveralls.io/github/atavory/ibex?branch=master

.. image:: http://readthedocs.org/projects/ibex/badge/?version=latest 
    :target: http://ibex.readthedocs.io/en/latest/?badge=latest

.. image:: doc/html/_static/logo.jpeg



`scikit-learn <http://scikit-learn.org/stable/>`_

`pandas <http://pandas.pydata.org/>`_

`documentation at readthedocs <http://ibex.readthedocs.io/en/latest/?badge=latest>`_


TL;DR
-----

    >>> import numpy as np
    >>> from sklearn import datasets
    >>> import pandas as pd

    >>> iris = datasets.load_iris()
    >>> iris = pd.DataFrame(
    ...     np.c_[iris['data'], iris['target']],
    ...     columns=iris['feature_names']+['class'])

    >>> iris.columns
    Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
           'petal width (cm)', 'class'],
          dtype='object')

A `pd.DataFrame` and 

    >>> x = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})                       
    >>> y = pd.Series([1, 2, 3])                                                                                                                                       

sdf
																					
    >>> prd = frame(preprocessing.StandardScaler()) | \                          
    ...     apply() + apply({'sqrt_a': np.sqrt, 'sqr_a': lambda x: x ** 2}, columns='a') | \
    ...     frame(linear_model.LinearRegression())                                  
    >>> y_hat = prd.fit(x, y).predict(x)   

API
---

    >>> from ibex import *

`FrameMixin`

