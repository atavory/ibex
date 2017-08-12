from __future__ import absolute_import


import inspect
import os
import types

import six
import numpy as np
import pandas as pd


_in_op_flag = '_ibex_xy_estimator_in_op_%s' % hash(os.path.abspath(__file__))


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
            if hasattr(self, _in_op_flag):
                return fn(X, *args, **kwargs)

            op_ind = ind[X[:, 0].astype(int)]
            X_ = pd.DataFrame(X[:, 1:], index=op_ind)

            base_attr = getattr(type(estimator), name)
            if six.PY3:
                params = list(inspect.signature(base_attr).parameters)
            else:
                params = inspect.getargspec(base_attr)[0]

            # Tmp Ami - write a ut for this; remove todo from docs
            # Tmp Ami - refactor this; it appears in _Adapter
            if len(params) > 2 and params[2] == 'y' and len(args) > 0 and args[0] is not None:
                args = list(args)[:]
                args[0] = pd.Series(args[0], index=op_ind)

            setattr(self, _in_op_flag, True)
            try:
                res = fn(X_, *args, **kwargs)
            finally:
                delattr(self, _in_op_flag)

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
