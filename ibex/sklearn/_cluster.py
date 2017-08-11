from __future__ import absolute_import


import functools
import inspect

import pandas as pd
from sklearn import base
from .._adapter import frame


def _wrap_transform_type(fn):
    @functools.wraps(fn)
    def wrapped(self, X, *args, **kwargs):
        ret = fn(self, X, *args, **kwargs)
        if isinstance(ret, pd.DataFrame):
            ret.columns = list(range(len(ret.columns)))
        return ret
    return wrapped


def _from_pickle(est, params):
    est = frame(est)

    _update_est(est)

    return est(**params)


def _update_est(est):
    est.transform = _wrap_transform_type(est.transform)
    est.fit_transform = _wrap_transform_type(est.fit_transform)
    est.__reduce__ = lambda self: (_from_pickle, (inspect.getmro(est)[1], self.get_params(deep=True), ))


_extra_doc = """

Example

    >>> import pandas as pd
    >>> import numpy as np
    >>> from ibex.sklearn import datasets
    >>> from ibex.sklearn.cluster import KMeans as PDKMeans

    >>> iris = datasets.load_iris()
    >>> features = iris['feature_names']
    >>> iris = pd.DataFrame(
    ...     np.c_[iris['data'], iris['target']],
    ...     columns=features+['class'])

    >>> iris[features]
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    0                5.1               3.5                1.4               0.2
    1                4.9               3.0                1.4               0.2
    2                4.7               3.2                1.3               0.2
    3                4.6               3.1                1.5               0.2
    4                5.0               3.6                1.4               0.2
    ...

    >>> clt = PDKMeans(n_clusters=3, random_state=1).fit(iris[features])
    >>> clt.transform(iris[features])
         0         1         2
    0    3.419251  0.146942  5.059542
    1    3.398574  0.438169  5.114943
    2    3.569357  0.412301  5.279355
    3    3.422410  0.518837  5.153590
    4    3.467264  0.197970  5.104334
    ...

"""


def update_module(name, module):
    module.__doc__ += _extra_doc

    for name in dir(module):
        c = getattr(module, name)
        try:
            if not issubclass(c, base.TransformerMixin):
                continue
        except TypeError:
            continue
        _update_est(c)
