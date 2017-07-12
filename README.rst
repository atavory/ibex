`frame_learn` - `pandas` adapters for `scikit-learn`
===================================================

Ami Tavory, Shahar Azulay, and Tali Raveh-Sadka

.. image:: https://travis-ci.org/atavory/frame_learn.svg?branch=master  
    :target: https://travis-ci.org/atavory/frame_learn

.. image:: https://landscape.io/github/atavory/frame_learn/master/landscape.svg?style=flat
   :target: https://landscape.io/github/atavory/frame_learn/master

.. image:: https://coveralls.io/repos/github/atavory/frame_learn/badge.png?branch=master
	:target: https://coveralls.io/github/atavory/frame_learn?branch=master


`scikit-learn <http://scikit-learn.org/stable/>`_

`pandas <http://pandas.pydata.org/>`_


TL;DR
-----

    >>> import pandas as pd
    >>> from sklearn import linear_model
    >>> from frame_learn import *

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

    >>> from frame_learn import *

`FrameMixin`

