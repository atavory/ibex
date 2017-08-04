from __future__ import absolute_import


import inspect

import six
import numpy as np
from sklearn import base


def _from_pickle(est, X, y):
    return make_xy_estimator(est, X, y)[0]


def make_xy_estimator(estimator, orig_X, orig_y=None):
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
        def fit_transform(self, X, *args):
            return self.__run(super(_Adapter, self).fit_transform, 'fit_transform', X, *args)

        def predict_proba(self, X, *args):
            return self.__run(super(_Adapter, self).predict_proba, 'predict_proba', X, *args)

        def sample_y(self, X, *args):
            return self.__run(super(_Adapter, self).sample_y, 'sample_y', X, *args)

        def score_samples(self, X, *args):
            return self.__run(super(_Adapter, self).score_samples, 'score_samples', X, *args)

        def staged_predict_proba(self, X, *args):
            return self.__run(super(_Adapter, self).staged_predict_proba, 'staged_predict_proba', X, *args)

        def apply(self, X, *args):
            return self.__run(super(_Adapter, self).apply, 'apply', X, *args)

        def bic(self, X, *args):
            return self.__run(super(_Adapter, self).bic, 'bic', X, *args)

        def perplexity(self, X, *args):
            return self.__run(super(_Adapter, self).perplexity, 'perplexity', X, *args)

        def fit(self, X, *args):
            return self.__run(super(_Adapter, self).fit, 'fit', X, *args)

        def decision_function(self, X, *args):
            return self.__run(super(_Adapter, self).decision_function, 'decision_function', X, *args)

        def aic(self, X, *args):
            return self.__run(super(_Adapter, self).aic, 'aic', X, *args)

        def partial_fit(self, X, *args):
            return self.__run(super(_Adapter, self).partial_fit, 'partial_fit', X, *args)

        def predict(self, X, *args):
            return self.__run(super(_Adapter, self).predict, 'predict', X, *args)

        def radius_neighbors(self, X, *args):
            return self.__run(super(_Adapter, self).radius_neighbors, 'radius_neighbors', X, *args)

        def staged_decision_function(self, X, *args):
            return self.__run(super(_Adapter, self).staged_decision_function, 'staged_decision_function', X, *args)

        def staged_predict(self, X, *args):
            return self.__run(super(_Adapter, self).staged_predict, 'staged_predict', X, *args)

        def inverse_transform(self, X, *args):
            return self.__run(super(_Adapter, self).inverse_transform, 'inverse_transform', X, *args)

        def fit_predict(self, X, *args):
            return self.__run(super(_Adapter, self).fit_predict, 'fit_predict', X, *args)

        def kneighbors(self, X, *args):
            return self.__run(super(_Adapter, self).kneighbors, 'kneighbors', X, *args)

        def predict_log_proba(self, X, *args):
            return self.__run(super(_Adapter, self).predict_log_proba, 'predict_log_proba', X, *args)

        def transform(self, X, *args):
            return self.__run(super(_Adapter, self).transform, 'transform', X, *args)

        def score(self, X, *args):
            return self.__run(super(_Adapter, self).score, 'score', X, *args)

        def __run(self, fn, name, X, *args):
            if hasattr(self, '_ibex_in_op'):
                return fn(X, *args)

            inds = X[:, 0]

            base_attr = getattr(type(estimator), name)
            if six.PY3:
                params = list(inspect.signature(base_attr).parameters)
            else:
                params = inspect.getargspec(base_attr)[0]

            # Tmp Ami - write a ut for this; remove todo from docs
            if len(params) > 2 and params[2] == 'y' and len(args) > 0 and args[0] is not None:
                args = list(args)[:]
                args[0] = orig_y.ix[inds]

            self._ibex_in_op = True
            try:
                res = fn(orig_X.ix[inds], *args)
            finally:
                delattr(self, '_ibex_in_op')

            return res

        def __reduce__(self):
            return (_from_pickle, (estimator, orig_X, orig_y))

        @property
        def orig_estimator(self):
            est = base.clone(estimator)
            return est.set_params(**get_set_params(self))


    n = len(orig_X)
    X_ = np.arange(n).reshape((n, 1))
    y_ = None if orig_y is None else orig_y.values

    return _Adapter(**get_set_params(estimator)), X_, y_
