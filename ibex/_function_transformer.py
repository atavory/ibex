from __future__ import absolute_import

import pandas as pd
from sklearn import base
from six import string_types

from ._frame_mixin import FrameMixin


__all__ = []


def _process_cols(cols):
    if cols is None:
        return None

    return [cols] if isinstance(cols, string_types) else list(cols)


# Tmp Ami - add kw_args, inverse shit
class _FunctionTransformer(base.BaseEstimator, base.TransformerMixin, FrameMixin):
    def __init__(self, func, in_cols, out_cols, pass_y, kw_args):
        FrameMixin.__init__(self)

        params = {
            'func': func,
            'in_cols': in_cols,
            'out_cols': out_cols,
            'pass_y': pass_y,
            'kw_args': kw_args,
        }

        self.set_params(**params)

    def fit(self, X, y=None):
        self.x_columns = X.columns

        if self.in_cols is not None:
            Xt = X[self.in_cols]
        else:
            Xt = X

        if self.func is None:
            return self

        if self.pass_y:
            self.func.fit(Xt, y)
        else:
            self.func.fit(Xt)

        return self

    def fit_transform(self, X, y=None):
        self.x_columns = X.columns

        Xt = X[self.x_columns]

        in_cols = _process_cols(self.in_cols)

        if in_cols is not None:
            Xt = Xt[in_cols]

        if self.func is None:
            res = Xt
        elif FrameMixin.is_subclass(self.func):
            if self.pass_y:
                res = self.func.fit_transform(Xt, y)
            else:
                res = self.func.fit_transform(Xt)
        else:
            res = pd.DataFrame(self.func(Xt), index=Xt.index)

        return self.__process_res(Xt, res)

    def transform(self, X, y=None):
        Xt = X[self.x_columns]

        in_cols = _process_cols(self.in_cols)

        if in_cols is not None:
            Xt = Xt[in_cols]

        if self.func is None:
            res = Xt
        elif FrameMixin.is_subclass(self.func):
            if self.pass_y:
                res = self.func.transform(Xt, y)
            else:
                res = self.func.transform(Xt)
        else:
            res = pd.DataFrame(self.func(Xt), index=Xt.index)

        return self.__process_res(Xt, res)

    def __process_res(self, Xt, res):
        in_cols = _process_cols(self.in_cols)
        out_cols = _process_cols(self.out_cols)
        if out_cols is not None:
            res_cols = out_cols
        elif in_cols is not None:
            res_cols = in_cols
        else:
            res_cols = Xt.columns

        res = res.copy()
        res.columns = res_cols
        return res

_FunctionTransformer.__name__ = 'FunctionTransformer'



