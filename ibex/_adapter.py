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

        def set_params(self, **params):

            if 'columns' in params:
                FrameMixin.set_params(self, columns=params['columns'])
                params = params.copy()
                del params['columns']

            super(_Adapter, self).set_params(**params)

        def get_params(self, deep=True):
            mixin_params = FrameMixin.get_params(self, deep=deep)
            wrapped_params = super(_Adapter, self).get_params(deep=deep)
            mixin_params.update(wrapped_params)
            return mixin_params

        def fit(self, X, *args):
            self.set_params(columns=X.columns)

            res = super(_Adapter, self).fit(X, *args)

            return self.__process_wrapped_call_res(X, res)

        def predict(self, X, *args):
            res = super(_Adapter, self).predict(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[self.get_params()['columns']], res)

        def transform(self, X, *args):
            res = super(_Adapter, self).transform(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[self.get_params()['columns']], res)

        # Tmp Ami - should be in base?
        def __x(self, X):
            # Tmp Ami - should be in base?
            X = X[self.get_params()['columns']]
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
