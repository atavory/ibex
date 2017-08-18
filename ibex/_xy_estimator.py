from __future__ import absolute_import


import inspect
import os
import types

import six
import numpy as np
import pandas as pd

from ._utils import get_wrapped_y, update_wrapped_y
from ._base import InOpChecker


_in_ops = InOpChecker(__file__)


def make_estimator(estimator, ind):
    ind = ind.copy()

    def get_set_params(est):
        base_attr = getattr(estimator, '__init__')
        if six.PY3:
            args = list(inspect.signature(base_attr).parameters)
        else:
            args = inspect.getargspec(base_attr)[0]
        orig_args = est.get_params()
        args = {arg: orig_args[arg] for arg in args if arg in orig_args}
        return args

    class _Adapter(type(estimator)):
        def fit(self, X, *args, **kwargs):
            self.__run(super(_Adapter, self).fit, 'fit', X, *args, **kwargs)
            return self

        def predict(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).predict, 'predict', X, *args, **kwargs)

        def fit_predict(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).fit_predict, 'fit_predict', X, *args, **kwargs)

        def __run(self, fn, name, X, *args, **kwargs):
            if self in _in_ops:
                return fn(X, *args, **kwargs)

            op_ind = ind[X[:, 0].astype(int)]
            X_ = pd.DataFrame(X[:, 1:], index=op_ind)

            y = get_wrapped_y(name, args)
            if y is not None:
                args = list(args)[:]
                if len(y.shape) == 1:
                    update_wrapped_y(args, pd.Series(y, index=op_ind))
                else:
                    update_wrapped_y(args, pd.DataFrame(y, index=op_ind))

            _in_ops.add(self)
            try:
                res = fn(X_, *args, **kwargs)
            finally:
                _in_ops.remove(self)

            return self.__process_wrapped_call_res(res)

        def __process_wrapped_call_res(self, res):
            if isinstance(res, pd.Series):
                return res.as_matrix()

            if isinstance(res, pd.DataFrame):
                return res.values

            if isinstance(res, types.GeneratorType):
                return (self.__process_wrapped_call_res(r) for r in res)

            return res

    return _Adapter(**get_set_params(estimator))


def make_xy(X, y):
    X_ = np.c_[range(len(X)), X.as_matrix()]
    y_ = y.values if y is not None else None
    return X_, y_
