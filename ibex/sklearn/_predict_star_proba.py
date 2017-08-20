from __future__ import absolute_import


from sklearn import base

from ._utils import get_matching_estimators


_extra_doc = """

.. tip::

    This module contains classifiers. The ``predict_proba``  and
    ``predict_log_proba`` methods for Ibex classifiers, return
    :mod:`pandas.DataFrame` objects whose columns are the classes.

    Example:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from ibex.sklearn import datasets
    >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression
    >>>
    >>> iris = datasets.load_iris()
    >>> features = iris['feature_names']
    >>> iris = pd.DataFrame(
    ...     np.c_[iris['data'], iris['target']],
    ...     columns=features+['class'])

    >>> clf = PdLogisticRegression().fit(iris[features], iris['class'])
    >>> clf.classes_
    array([ 0.,  1.,  2.])

    >>> clf.predict_proba(iris[features])
             0...      1...          2...
    0    0.879682  0.120308  1.081314e-05
    1    0.799706  0.200263  3.038254e-05
    2    0.853797  0.146177  2.590313e-05
    3    0.825383  0.174559  5.793567e-05
    4    0.897324  0.102665  1.120500e-05
    5    0.926987  0.073000  1.296939e-05
     ...

    >>> clf.predict_log_proba(iris[features])
              0...      1...       2...
    0    -0.128195 -2.117704 -11.434749
    1    -0.223511 -1.608122 -10.401643
    2    -0.158062 -1.922935 -10.561147
    3    -0.191908 -1.745493  -9.756177
    4    -0.108339 -2.276282 -11.399150
    5    -0.075816 -2.617290 -11.252919
    ...

"""


def update_module(module):
    if get_matching_estimators(module, base.TransformerMixin):
        module.__doc__ += _extra_doc





