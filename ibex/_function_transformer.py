import pandas as pd
from six import string_types

from ._frame_mixin import FrameMixin


__all__ = []


# Tmp Ami - add kw_args, inverse shit
class _FunctionTransformer(FrameMixin):
    def __init__(self, func, pass_y, kw_args, columns, trans_columns):
        FrameMixin.__init__(self)

        self._func, self._pass_y, self._kw_args, self._cols, self._trans_cols = \
            func, pass_y, kw_args, columns, trans_columns

    def fit(self, x, y=None):
        x = self._prep_x(x)

        if not FrameMixin.is_subclass(self._func):
            return

        if self._pass_y:
            self._func.fit(x, y)
        else:
            self._func.fit(x)

        return self

    def fit_transform(self, x, y=None):
        x = self._prep_x(x)

        if self._func is None:
            return x

        if not FrameMixin.is_subclass(self._func):
            return self._func(x, y) if self._pass_y else self._func(x)

        if self._pass_y:
            res = self._func.fit_transform(x, y)
        else:
            res = self._func.fit_transform(x)

        return self._process_trans_res(res)

    def _prep_x(self, x):
        if self._cols is None:
            return x

        columns = \
            [self._cols] if isinstance(self._cols, string_types) else self._cols
        return x[columns]

    def transform(self, x, y=None):
        x = self._prep_x(x)

        if self._func is None:
            return x

        if not FrameMixin.is_subclass(self._func):
            res = self._func(x, y) if self._pass_y else self._func(x)
        elif self._pass_y:
            res = self._func.fit_transform(x, y)
        else:
            res = self._func.fit_transform(x)

        return self._process_trans_res(res)

    # Tmp Ami - set_params? ut?
    def get_params(self, deep=True):
        return {
            'func': self._func,
            'pass_y': self._pass_y,
            'kw_args': self._kw_args,
            'columns': self._cols,
            'trans_columns': self._trans_cols}

    def _process_trans_res(self, res):
        if self._trans_cols is not None:
            res.columns = self._trans_cols
        return res


def trans(func=None, pass_y=False, kw_args=None, columns=None, trans_columns=None):
    return _FunctionTransformer(func, pass_y, kw_args, columns, trans_columns)

__all__ += ['trans']

