import inspect
import types
import functools
import copy

import six
import numpy as np
import pandas as pd
import sklearn

from ._frame_mixin import FrameMixin


def frame(step):
    class _Adapter(step, FrameMixin):
        def fit(self, X, *args):
            self.set_x(X)

            res = step.fit(self, X, *args)

            return self.__process_wrapped_call_res(X, res)

        def predict(self, X, *args):
            self.set_x(X)

            res = step.predict(self, X, *args)

            return self.__process_wrapped_call_res(X, res)

        def transform(self, X, *args):
            self.set_x(X)

            res = step.transform(self, X, *args)

            return self.__process_wrapped_call_res(X, res)

        def __x(self, X):
            return X if FrameMixin.is_subclass(step) else X.as_matrix()

        def __y(self, y):
            if y is None:
                return None
            return y if FrameMixin.is_subclass(step) else y.values

        def __process_wrapped_call_res(self, X, res):
            if isinstance(res, np.ndarray):
                if len(res.shape) == 1:
                    return pd.Series(res, index=X.index)
                if len(res.shape) == 2:
                    if len(X.columns) == res.shape[1]:
                        columns = X.columns
                    else:
                        columns = range(res.shape[1])
                    return pd.DataFrame(res, index=X.index, columns=columns)

            return res

    _Adapter.__name__ = step.__name__
    _Adapter.__doc__ = step.__doc__

    return _Adapter
