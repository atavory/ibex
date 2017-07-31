"""
Auto-generated :mod:`ibex.sklearn` wrapper for :mod:`sklearn.linear_model`.
"""

from __future__ import absolute_import


import inspect

from sklearn import feature_selection as _orig
from sklearn import base

import ibex


for name in dir(_orig):
    if name.startswith('_'):
        continue
    est = getattr(_orig, name)
    try:
        if inspect.isclass(est) and issubclass(est, base.BaseEstimator):
            globals()[name] = ibex.frame(est)
        else:
            globals()[name] = est
    except TypeError as e:
        pass
