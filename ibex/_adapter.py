from __future__ import absolute_import

import inspect
import types

import six
import numpy as np
import pandas as pd
from sklearn import pipeline

from ._frame_mixin import FrameMixin


def frame(step):
    """
    Arguments:
        step: blah

    Returns:
        `step` :py:class:`sklearn.base.BaseEstimator` :py:class:`ibex.FrameMixin`
    """
    if isinstance(step, pipeline.Pipeline):
        return frame(pipeline.Pipeline)(steps=step.steps)

    if not inspect.isclass(step):
        params = step.get_params()
        f = frame(type(step))(**params)
        return f

    class _Adapter(step, FrameMixin):
        def __repr__(self):
            return step.__repr__(self).replace('_Adapter', 'Adapter[' + step.__name__ + ']', 1)

        def __str__(self):
            return step.__str__(self).replace('_Adapter', 'Adapter[' + step.__name__ + ']', 1)

        def fit(self, X, *args):
			return self.__run(self, super(_Adapter, self).fit, True, X, *args)

        def predict(self, X, *args):
            res = super(_Adapter, self).predict(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[self.x_columns], res)

        def fit_transform(self, X, *args):
            self.x_columns = X.columns

            res = super(_Adapter, self).fit_transform(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[self.x_columns], res)

        def transform(self, X, *args):
            res = super(_Adapter, self).transform(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[self.x_columns], res)

		def __run(self, fn, fit, X, *args):
            # Tmp Ami - why not in function adapter? where are uts?
			if fit:
				self.x_columns = X.columns

            res = fn.fit(self.__x(X), *args)

            return self.__process_wrapped_call_res(X, res)

        # Tmp Ami - should be in base?
        def __x(self, X):
            # Tmp Ami - should be in base?
            X = X[self.x_columns]
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
                        columns = [' ' for _ in range(res.shape[1])]
                    return pd.DataFrame(res, index=X.index, columns=columns)

            return res

    _Adapter.__name__ = step.__name__

    for name, func in vars(_Adapter).items():
        if name.startswith('_'):
            continue

        if not six.callable(func) and not func.__doc__:
            continue

        parfunc = getattr(step, name, None)
        if parfunc and getattr(parfunc, '__doc__', None):
            func.__doc__ = parfunc.__doc__

    if not hasattr(step, 'fit_transform') and hasattr(_Adapter, 'fit_transform'):
        delattr(_Adapter, 'fit_transform')

    return _Adapter
