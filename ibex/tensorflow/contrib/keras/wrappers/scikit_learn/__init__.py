"""
Wrappers for :mod:`tensorflow.contrib.keras.wrappers.sklearn`.

.. versionadded:: 1.2

"""


from __future__ import absolute_import

import functools

import pandas as pd
from sklearn import base
from tensorflow.contrib import keras

from ......_adapter import frame_ex
from ......_base import FrameMixin, verify_x_type, verify_y_type


class _KeraseEstimator(base.BaseEstimator, FrameMixin):
    def fit(self, X, y, **fit_params):
        verify_x_type(X)
        verify_y_type(y)
        self.x_columns = X.columns
        # Tmp Ami - should go in utils
        if y is not None and not X.index.equals(y.index):
            raise ValueError('Indexes do not match')
        uX = self._x(False, X)
        uy = self._y(y)
        self.history_ = self._fit(X, uX, y, uy, **fit_params)
        return self

    def predict(self, X):
        verify_x_type(X)
        uX = self._x(False, X)
        return self._predict(X, uX)

    def score(self, X, y, **kwargs):
        verify_x_type(X)
        verify_y_type(y)
        # Tmp Ami - should go in utils
        if y is not None and not X.index.equals(y.index):
            raise ValueError('Indexes do not match')
        uX = self._x(False, X)
        uy = self._y(y)
        return self._score(X, uX, y, uy, **kwargs)

    def _x(self, inv, X):
        return X[self.x_columns].as_matrix() if not inv else X.as_matrix()

    def aic(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).aic,
            'aic',
            X,
            *args,
            **kwargs)

    def apply(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).apply,
            'apply',
            X,
            *args,
            **kwargs)

    def decision_function(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).decision_function,
            'decision_function',
            X,
            *args,
            **kwargs)

    def bic(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).bic,
            'bic',
            X,
            *args,
            **kwargs)

    def inverse_transform(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).inverse_transform,
            'inverse_transform',
            X,
            *args,
            **kwargs)

    def kneighbors(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).kneighbors,
            'kneighbors',
            X,
            *args,
            **kwargs)

    def partial_fit(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).partial_fit,
            'partial_fit',
            X,
            *args,
            **kwargs)

    def radius_neighbors(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).radius_neighbors,
            'radius_neighbors',
            X,
            *args,
            **kwargs)

    def score_samples(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).score_samples,
            'score_samples',
            X,
            *args,
            **kwargs)

    def staged_decision_function(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).staged_decision_function,
            'staged_decision_function',
            X,
            *args,
            **kwargs)

    def staged_predict(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).staged_predict,
            'staged_predict',
            X, *args,
            **kwargs)

    def transform(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).transform,
            'transform',
            X,
            *args,
            **kwargs)

    def __adapter_run(self, fn, name, X, *args, **kwargs):
        if self in _in_ops:
            return fn(X, *args, **kwargs)

        if not isinstance(X, pd.DataFrame):
            verify_x_type(X)

        # Tmp Ami - why not in function adapter? where are uts?
        if name.startswith('fit'):
            self.x_columns = X.columns

        y = get_wrapped_y(name, args)
        verify_y_type(y)
        if y is not None and not X.index.equals(y.index):
            raise ValueError('Indexes do not match')
        if y is not None:
            if name.startswith('fit'):
                self.y_columns = y.columns if isinstance(y, pd.DataFrame) else None

        inv = name == 'inverse_transform'

        _in_ops.add(self)
        try:
            res = fn(self.__x(inv, X), *args, **kwargs)
        finally:
            _in_ops.remove(self)

        ret = self.__adapter_process_wrapped_call_res(inv, X, res)

        if name in extra_methods_d:
            ret = extra_methods_d[name](self, ret)

        return ret

    def __adapter_process_wrapped_call_res(self, inv, X, res):
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

    def __reduce__(self):
        if not self.__module__.startswith('ibex'):
            raise TypeError('Cannot serialize a subclass of this type; please use composition instead')
        return (_from_pickle, (est, self.get_params(deep=True), extra_methods, extra_attribs, post_op))

    def _repr(self):
        params = ','.join('%s=%s' % (k, v) for (k, v) in self.get_params().items())
        return 'Adapter[' + type(self).__name__ + '](' + params + ')'

    def _str(self):
        return self._repr()


class KerasClassifier(
        _KeraseEstimator,
        keras.wrappers.scikit_learn.KerasClassifier):
    pass


class KerasRegressor(
        _KeraseEstimator,
        keras.wrappers.scikit_learn.KerasRegressor):

    __KerasBase = keras.wrappers.scikit_learn.KerasRegressor

    def __init__(self, build_fn=None, **sk_params):
        KerasRegressor.__KerasBase.__init__(self, build_fn, **sk_params)

    def __repr__(self):
        return _KeraseEstimator._repr(self)

    def __str__(self):
        return _KeraseEstimator._str(self)

    def _fit(self, X, uX, y, uy, **fit_params):
        return KerasRegressor.__KerasBase.fit(self, uX, uy, **fit_params)

    def _score(self, X, uX, y, uy, **kwargs):
        return KerasRegressor.__KerasBase.score(self, uX, uy, **kwargs)

    def _predict(self, X, uX):
        return pd.Series(
            KerasRegressor.__KerasBase.predict(self, uX),
            index=X.index)

    def aic(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).aic,
            'aic',
            X,
            *args,
            **kwargs)

    def apply(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).apply,
            'apply',
            X,
            *args,
            **kwargs)

    def decision_function(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).decision_function,
            'decision_function',
            X,
            *args,
            **kwargs)

    def bic(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).bic,
            'bic',
            X,
            *args,
            **kwargs)

    def inverse_transform(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).inverse_transform,
            'inverse_transform',
            X,
            *args,
            **kwargs)

    def kneighbors(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).kneighbors,
            'kneighbors',
            X,
            *args,
            **kwargs)

    def partial_fit(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).partial_fit,
            'partial_fit',
            X,
            *args,
            **kwargs)

    def radius_neighbors(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).radius_neighbors,
            'radius_neighbors',
            X,
            *args,
            **kwargs)

    def score_samples(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).score_samples,
            'score_samples',
            X,
            *args,
            **kwargs)

    def staged_decision_function(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).staged_decision_function,
            'staged_decision_function',
            X,
            *args,
            **kwargs)

    def staged_predict(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).staged_predict,
            'staged_predict',
            X, *args,
            **kwargs)

    def transform(self, X, *args, **kwargs):
        return self.__adapter_run(
            super(_Adapter, self).transform,
            'transform',
            X,
            *args,
            **kwargs)

    def __adapter_run(self, fn, name, X, *args, **kwargs):
        if self in _in_ops:
            return fn(X, *args, **kwargs)

        if not isinstance(X, pd.DataFrame):
            verify_x_type(X)

        # Tmp Ami - why not in function adapter? where are uts?
        if name.startswith('fit'):
            self.x_columns = X.columns

        y = get_wrapped_y(name, args)
        verify_y_type(y)
        if y is not None and not X.index.equals(y.index):
            raise ValueError('Indexes do not match')
        if y is not None:
            if name.startswith('fit'):
                self.y_columns = y.columns if isinstance(y, pd.DataFrame) else None

        inv = name == 'inverse_transform'

        _in_ops.add(self)
        try:
            res = fn(self.__x(inv, X), *args, **kwargs)
        finally:
            _in_ops.remove(self)

        ret = self.__adapter_process_wrapped_call_res(inv, X, res)

        if name in extra_methods_d:
            ret = extra_methods_d[name](self, ret)

        return ret

    def __adapter_process_wrapped_call_res(self, inv, X, res):
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

    def _y(self, y):
        return y.values


__all__ = ['KerasClassifier', 'KerasRegressor']

