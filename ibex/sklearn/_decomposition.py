from __future__ import absolute_import


import functools
import inspect

import pandas as pd
from sklearn import base
from .._adapter import  frame


def _wrap_transform_type(fn):
    @functools.wraps(fn)
    def wrapped(self, X, *args, **kwargs):
        ret = fn(self, X, *args, **kwargs)
        if isinstance(ret, pd.DataFrame):
            ret.columns = ['comp_%i' % i for i in range(len(ret.columns))]
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


def update_module(name, module):
    if name != 'decomposition':
        return

    for name in dir(module):
        c = getattr(module, name)
        try:
            if not issubclass(c, base.TransformerMixin):
                continue
        except TypeError:
            continue
        _update_est(c)





