from __future__ import absolute_import

import inspect
import types
import functools
import copy

import six
import numpy as np
import pandas as pd
from sklearn import base
from sklearn import pipeline

from ._frame_mixin import FrameMixin


def frame(step):
    if isinstance(step, pipeline.Pipeline):
        return frame(pipeline.Pipeline)(steps=step.steps)

    if not inspect.isclass(step):
        f = frame(type(step))()
        params = step.get_params()
        f.set_params(params)
        return f

    class _Adapter(step, FrameMixin):
        __name__ = step.__name__
        __doc__ = step.__doc__

        def fit(self, X, *args):
            FrameMixin.set_params(self, columns=X.columns)

            res = super(_Adapter, self).fit(X, *args)

            return self.__process_wrapped_call_res(X, res)

        def predict(self, X, *args):
            res = super(_Adapter, self).predict(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[FrameMixin.get_params(self)['columns']], res)

        def transform(self, X, *args):
            res = super(_Adapter, self).transform(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[FrameMixin.get_params(self)['columns']], res)

        # Tmp Ami - should be in base?
        def __x(self, X):
            # Tmp Ami - should be in base?
            X = X[FrameMixin.get_params(self)['columns']]
            # Tmp Ami - is_subclass or isinstance?
            return X if FrameMixin.is_subclass(self) else X.as_matrix()

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

    return _Adapter
