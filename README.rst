`ibex` - `pandas` adapters for `scikit-learn`
===================================================

Ami Tavory, Shahar Azulay, and Tali Raveh-Sadka

.. image:: https://travis-ci.org/atavory/ibex.svg?branch=master  
    :target: https://travis-ci.org/atavory/ibex

.. image:: https://landscape.io/github/atavory/ibex/master/landscape.svg?style=flat
   :target: https://landscape.io/github/atavory/ibex/master

.. image:: https://coveralls.io/repos/github/atavory/ibex/badge.png?branch=master
	:target: https://coveralls.io/github/atavory/ibex?branch=master

.. image:: https://readthedocs.org/projects/ibex/badge/
    :target: http://ibex.readthedocs.io/en/latest/?badge=latest

.. image:: https://www.workitdaily.com/wp-content/uploads/2012/12/incomplete-degree-resume.jpg



`scikit-learn <http://scikit-learn.org/stable/>`_

`pandas <http://pandas.pydata.org/>`_

`documentation <https://atavory.github.io/ibex/>`_


TL;DR
-----

    >>> import pandas as pd
    >>> from sklearn import linear_model
    >>> from ibex import *

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

