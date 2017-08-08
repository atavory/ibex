from __future__ import absolute_import


import functools

import pandas as pd
from sklearn import base


def _wrap_transform_type(fn):
    @functools.wraps(fn)
    def wrapped(self, X, *args, **kwargs):
        ret = fn(self, X, *args, **kwargs)
        if isinstance(ret, pd.DataFrame):
            ret.columns = X.columns[self.get_support(indices=True)]
        return ret
    return wrapped


def update_module(name, module):
    if name != 'feature_selection':
        return

    for _ in range(20):
        for name in dir(module):
            c = getattr(module, name)
            try:
                if not issubclass(c, base.TransformerMixin):
                    continue
            except TypeError:
                continue
            if not hasattr(c, 'get_support'):
                continue
            c.transform = _wrap_transform_type(c.transform)
            c.fit_transform = _wrap_transform_type(c.fit_transform)





