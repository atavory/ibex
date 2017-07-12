import pandas as pd
from six import string_types

from ._frame_mixin import FrameMixin


class _FunctionTransformer(FrameMixin):
    """
    Applies some step to only some (or one) columns - Pandas version.

    Arguments:
        wh: Something for which df[:, wh]
            is defined, where df is a pandas DataFrame.
        st: A step.

    Example:

        import day_two

        x = np.linspace(-3, 3, 50)
        y = x
        # Apply the step only to the first column of h.
        sm = day_two.sklearn_.preprocessing\
            .FunctionTransformer(
                'moshe',
                day_two.preprocessing.UnivariateSplineSmoother())
        sm.fit(
            pd.DataFrame({'moshe': x, 'koko': x}),
            pd.Series(y))
    """
    def __init__(self, func, pass_y, kw_args, columns):
        FrameMixin.__init__(self)

        self._func, self._pass_y, self._kw_args, self._columns = \
            func, pass_y, kw_args, columns

    def fit(self, x, y, **fit_params):
        # Tmp AmiAdd here call to fit
        return self

    def transform(self, x, y=None):
        # Tmp Ami Add here call to fit
        if self._columns is not None:
            # Tmp AMi - refactor next to utility in top of file
            columns = \
                [self._columns] if isinstance(self._columns, string_types) else self._columns
            x = x[columns]

        if self._func is None:
            return x

        if isinstance(self._func, dict):
            dfs = []
            for k, v in self._func.items():
                res = pd.DataFrame(v(x))
                columns = [k] if isinstance(k, string_types) else k
                res.columns = columns
                dfs.append(res)
            return pd.concat(dfs, axis=1)

        return self._func(x)

    # Tmp Ami - add fit_transform


def trans(func=None, pass_y=False, kw_args=None, columns=None):
    return _FunctionTransformer(func, pass_y, kw_args, columns)
