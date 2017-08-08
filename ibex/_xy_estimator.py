from __future__ import absolute_import


import inspect
import os

import six
import numpy as np
import pandas as pd
from sklearn import base


_in_op_flag = '_ibex_in_op_%s' % hash(os.path.abspath(__file__))


def _from_pickle(est, X, y, output_arrays):
    return make_estimator(est, ind, output_arrays)[0]


def make_estimator(estimator, ind, output_arrays=False):
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
        def fit_transform(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).fit_transform, 'fit_transform', X, *args, **kwargs)

        def predict_proba(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).predict_proba, 'predict_proba', X, *args, **kwargs)

        def sample_y(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).sample_y, 'sample_y', X, *args, **kwargs)

        def score_samples(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).score_samples, 'score_samples', X, *args, **kwargs)

        def staged_predict_proba(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).staged_predict_proba, 'staged_predict_proba', X, *args, **kwargs)

        def apply(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).apply, 'apply', X, *args, **kwargs)

        def bic(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).bic, 'bic', X, *args, **kwargs)

        def perplexity(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).perplexity, 'perplexity', X, *args, **kwargs)

        def fit(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).fit, 'fit', X, *args, **kwargs)

        def decision_function(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).decision_function, 'decision_function', X, *args, **kwargs)

        def aic(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).aic, 'aic', X, *args, **kwargs)

        def partial_fit(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).partial_fit, 'partial_fit', X, *args, **kwargs)

        def predict(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).predict, 'predict', X, *args, **kwargs)

        def radius_neighbors(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).radius_neighbors, 'radius_neighbors', X, *args, **kwargs)

        def staged_decision_function(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).staged_decision_function, 'staged_decision_function', X, *args, **kwargs)

        def staged_predict(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).staged_predict, 'staged_predict', X, *args, **kwargs)

        def inverse_transform(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).inverse_transform, 'inverse_transform', X, *args, **kwargs)

        def fit_predict(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).fit_predict, 'fit_predict', X, *args, **kwargs)

        def kneighbors(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).kneighbors, 'kneighbors', X, *args, **kwargs)

        def predict_log_proba(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).predict_log_proba, 'predict_log_proba', X, *args, **kwargs)

        def transform(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).transform, 'transform', X, *args, **kwargs)

        def score(self, X, *args, **kwargs):
            return self.__run(super(_Adapter, self).score, 'score', X, *args, **kwargs)

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
                delattr(self,_in_op_flag)

            return self.__process_wrapped_call_res(res)

        def __reduce__(self):
            return (_from_pickle, (estimator, ind, output_arrays))

        @property
        def orig_estimator(self):
            est = base.clone(estimator)
            return est.set_params(**get_set_params(self))

        def __process_wrapped_call_res(self, res):
            if not output_arrays:
                return res

            if isinstance(res, pd.Series):
                return res.as_matrix()

            if isinstance(res, pd.DataFrame):
                return res.values

            return res

    return _Adapter(**get_set_params(estimator))


def make_xy(X, y):
    X_ = np.c_[range(len(X)), X.as_matrix()]
    y_ = y.values if y is not None else None
    return X_, y_
