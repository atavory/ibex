"""
Wrappers for :mod:`tensorflow.contrib.keras.wrappers.sklearn`.

.. versionadded:: 1.2

"""


from __future__ import absolute_import

import types

import numpy as np
import pandas as pd
from sklearn import base
from tensorflow.contrib import keras
from sklearn import metrics

from ......_adapter import frame_ex
from ......_base import FrameMixin, verify_x_type, verify_y_type


class KerasEstimator(base.BaseEstimator, FrameMixin):
    def __init__(self, build_fn, cls, **sk_params):
        self._build_fn, self._cls, self._sk_params = build_fn, cls, sk_params

    def __repr__(self):
        params = ','.join('%s=%s' % (k, v) for (k, v) in self.get_params().items())
        return 'Adapter[' + type(self).__name__ + '](' + params + ')'

    def __str__(self):
        return self.__repr__()

    def get_params(self, deep=False):
        d = self._sk_params.copy()
        d['build_fn'] = self._build_fn
        return d

    def set_params(self, **sk_params):
        self._build_fn = sk_params['build_fn']

    # Tmp Ami
    # Use this http://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimatorhttp://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator

    def _x(self, inv, X):
        return X[self.x_columns].as_matrix() if not inv else X.as_matrix()


class KerasClassifier(KerasEstimator, base.ClassifierMixin):
    def __init__(self, build_fn, classes, **sk_params):
        KerasEstimator.__init__(self, build_fn, keras.wrappers.scikit_learn.KerasClassifier, **sk_params)
        self._classes = classes

    def get_params(self, deep=False):
        d = KerasEstimator.get_params(self, deep).copy()
        d['classes'] = self._classes
        return d

    def set_params(self, **sk_params):
        KerasEstimator.set_params(**sk_params)
        self._classes = sk_params['classes']

    def fit(self, X, y, **fit_params):
        # Tmp Ami
        np.random.seed(7)
        verify_x_type(X)
        verify_y_type(y)
        self.x_columns = X.columns
        # Tmp Ami - should go in utils
        if y is not None and not X.index.equals(y.index):
            raise ValueError('Indexes do not match')
        uX = self._x(False, X)
        uy = self._y(y)
        self._y_columns = uy.columns
        self._est = keras.wrappers.scikit_learn.KerasRegressor(self._build_fn, **self._sk_params)
        self.history_ = self._est.fit(uX, uy.values, **fit_params)
        return self

    def predict(self, X):
        verify_x_type(X)
        uX = self._x(False, X)
        res = self._est.predict(uX)
        res = pd.DataFrame(res, index=X.index, columns=self._y_columns).idxmax(axis=1)
        return res

    def predict_proba(self, X, *args, **kwargs):
        verify_x_type(X)
        uX = X[self.x_columns]
        res = self._est.predict(uX.values)
        return pd.DataFrame(res, index=X.index, columns=self._y_columns)

    def _y(self, y):
        dummies = pd.get_dummies(y)
        d = {k: dummies[k] if k in dummies.columns else 0 for k in self._classes}
        df = pd.DataFrame(d)[list(self._classes)]
        return df


class KerasRegressor(KerasEstimator, base.RegressorMixin):
    def __init__(self, build_fn, **sk_params):
        KerasEstimator.__init__(self, build_fn, keras.wrappers.scikit_learn.KerasRegressor, **sk_params)

    def fit(self, X, y, **fit_params):
        verify_x_type(X)
        verify_y_type(y)
        self.x_columns = X.columns
        # Tmp Ami - should go in utils
        if y is not None and not X.index.equals(y.index):
            raise ValueError('Indexes do not match')
        uX = self._x(False, X)
        uy = self._y(y)
        self._est = keras.wrappers.scikit_learn.KerasRegressor(self._build_fn, **self._sk_params)
        self.history_ = self._est.fit(uX, uy, **fit_params)
        return self

    def predict(self, X):
        verify_x_type(X)
        uX = self._x(False, X)
        res = self._est.predict(uX)
        return self._process_wrapped_call_res(False, X, res)

    def score(self, X, y, **kwargs):
        verify_x_type(X)
        verify_y_type(y)
        # Tmp Ami - should go in utils
        if y is not None and not X.index.equals(y.index):
            raise ValueError('Indexes do not match')
        uX = self._x(False, X)
        uy = self._y(y)
        res = self._est.score(uX, uy)
        return res

    def _y(self, y):
        return y.values

    def _process_wrapped_call_res(self, inv, X, res):
        if inv:
            return pd.DataFrame(res, index=X.index, columns=self.x_columns)

        X = X[self.x_columns]

        if isinstance(res, np.ndarray):
            if len(res.shape) == 1:
                return pd.Series(res, index=X.index)

            if len(res.shape) == 2:
                if len(X.columns) == res.shape[1]:
                    columns = X.columns
                else:
                    columns = [' ' for _ in range(res.shape[1])]
                return pd.DataFrame(res, index=X.index, columns=columns)

        if isinstance(res, types.GeneratorType):
            return (self.__adapter_process_wrapped_call_res(False, X, r) for r in res)

        return res

__all__ = ['KerasClassifier', 'KerasRegressor']

