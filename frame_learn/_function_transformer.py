import pandas as pd
from six import string_types

from ._frame_mixin import FrameMixin


# Tmp Ami - add kw_args, inverse shit
class _FunctionTransformer(FrameMixin):
    def __init__(self, func, pass_y, kw_args, columns):
        FrameMixin.__init__(self)

        self._func, self._pass_y, self._kw_args, self._columns = \
            func, pass_y, kw_args, columns

    def fit(self, x, y=None):
        x = self._prep_x(x)

        if not isinstance(self._func, dict):
            self._single_fit(self._func, x, y)
            return self

        for k, v in self._func.items():
            self._single_fit(v, x, y)

        return self

    def _single_fit(self, stage, x, y):
        if not FrameMixin.is_subclass(stage):
            return

        if self._pass_y:
            stage.fit(x, y)
        else:
            stage.fit(x)

    def fit_transform(self, x, y=None):
        x = self._prep_x(x)

        if not isinstance(self._func, dict):
            return self._single_fit_transform(self._func, x, y)

        dfs = []
        for k, v in self._func.items():
            res = pd.DataFrame(self._single_fit_transform(v, x, y))
            columns = [k] if isinstance(k, string_types) else k
            res.columns = columns
            dfs.append(res)
        return pd.concat(dfs, axis=1)

    def _single_fit_transform(self, stage, x, y):
        if stage is None:
            return x

        if not FrameMixin.is_subclass(stage):
            return stage(x, y) if self._pass_y else stage(x)

        # Tmp Ami - bad
        return stage.fit_transform(x, y) if self._pass_y else stage.fit_transform(x)

    def _prep_x(self, x):
        if self._columns is None:
            return x

        columns = \
            [self._columns] if isinstance(self._columns, string_types) else self._columns
        return x[columns]

    def transform(self, x, y=None):
        x = self._prep_x(x)

        if not isinstance(self._func, dict):
            return self._single_transform(self._func, x, y)

        dfs = []
        for k, v in self._func.items():
            res = pd.DataFrame(self._single_transform(v, x, y))
            columns = [k] if isinstance(k, string_types) else k
            res.columns = columns
            dfs.append(res)
        return pd.concat(dfs, axis=1)

    def _single_transform(self, stage, x, y):
        if stage is None:
            return x

        if not FrameMixin.is_subclass(stage):
            return stage(x, y) if self._pass_y else stage(x)

        return stage.transform(x, y) if self._pass_y else stage.transform(x)


def trans(func=None, pass_y=False, kw_args=None, columns=None):
    return _FunctionTransformer(func, pass_y, kw_args, columns)
