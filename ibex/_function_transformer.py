from __future__ import absolute_import

import pandas as pd
from sklearn import base
from six import string_types

from ._frame_mixin import FrameMixin


__all__ = []


def _flatten(func):
    if not isinstance(func, dict):
        return [(func, None, None)]

    ret = []
    for k, v in func.items():
        if not isinstance(v, dict):
            ret.append((v, _to_list(k), None))
            continue

        for kk, vv in v.items():
            ret.append((vv, _to_list(k), _to_list(kk)))
    return ret

def _to_list(cols):
    if cols is None:
        return None
    return [cols] if isinstance(cols, string_types) else cols


# Tmp Ami - add kw_args, inverse shit
class _FunctionTransformer(base.BaseEstimator, base.TransformerMixin, FrameMixin):
    def __init__(self, func, pass_y, kw_args):
        FrameMixin.__init__(self)

        params = {
            'func': func,
            'pass_y': pass_y,
            'kw_args': kw_args,
        }

        self.set_params(**params)

    def fit(self, x, y=None):
        # Tmp Ami - set x? uts?
        for t in self._flattened:
            func, cols = t[0], t[1]

            if func is None:
                continue

            if columns is not None:
                x = x[cols]

            if self._pass_y:
                t[0].fit(x, y)
            else:
                t[0].fit(x)

    def fit_transform(self, x, y=None):
        dfs = []

        for t in self._flattened:
            func, columns, out_columns = t[0], t[1], t[2]

            if columns is not None:
                x = x[cols]

            if func is None:
                df = x
            elif FrameMixin.is_subclass(t[0]):
                if self._pass_y:
                    df = t[0].fit_transform(x, y)
                else:
                    df = t[0].fit_transform(x)
            else:
                # Tmp Ami
                ff

        return pd.concat(dfs, axis=1)

    def transform(self, x, y=None):
        dfs = []

        for t in self._flattened:
            func, columns, out_columns = t[0], t[1], t[2]

            if columns is not None:
                x = x[cols]

            if func is None:
                df = x
            elif FrameMixin.is_subclass(t[0]):
                if self._pass_y:
                    df = t[0].transform(x, y)
                else:
                    df = t[0].transform(x)
            else:
                # Tmp Ami
                ff

        return pd.concat(dfs, axis=1)

    def set_params(self, **params):
        base.BaseEstimator.set_params(self, **params)

        self._flattened = _flatten(self.get_params()['func'])


def trans(func=None, pass_y=False, kw_args=None):
    return _FunctionTransformer(func, pass_y, kw_args)

__all__ += ['trans']

