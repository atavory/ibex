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
import six

from ......_base import FrameMixin, verify_x_type, verify_y_type
from ......_utils import update_method_wrapper, update_class_wrapper
from ......_utils import wrapped_fn_names


class KerasEstimator(base.BaseEstimator, FrameMixin):
    def __init__(self, build_fn, **sk_params):
        self._build_fn, self._sk_params = build_fn, sk_params

    # Tmp Ami - find how to show in sphinx
    @property
    def history_(self):
        """
        Returns:

            A :class:`tensorflow.contrib.keras.wrappers.scikit_learn` object describing the fit.

        Note:

            Should not be used before a call to ``fit``.
        """
        return self._history

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

    def _x(self, inv, X):
        return X[self.x_columns].as_matrix() if not inv else X.as_matrix()


class KerasClassifier(KerasEstimator, base.ClassifierMixin):
    """
    A tensorflow/keras classifier.

    Arguments:

        build_fn: Function returning a :class:`tensorflow.contrib.Model` object.
        classes: All possible values of the classes.

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.tensorflow.contrib.keras.wrappers.scikit_learn import KerasClassifier as PdKerasClassifier
        >>> import tensorflow
        ...
        ...
        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])
        ...
        >>> def _build_classifier_nn():
        ...     model = tensorflow.contrib.keras.models.Sequential()
        ...     model.add(tensorflow.contrib.keras.layers.Dense(8, input_dim=4, activation='relu'))
        ...     model.add(tensorflow.contrib.keras.layers.Dense(3, activation='softmax'))
        ...     model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        ...     return model
        ...
        >>> clf = PdKerasClassifier(
        ...     build_fn=_build_classifier_nn,
        ...     classes=iris['class'].unique(),
        ...     verbose=0)
        >>> clf.fit(iris[features], iris['class'])
        Adapter[KerasClassifier](...classes=[ 0.  1.  2.]...)
        >>> clf.history_
        <tensorflow.contrib.keras.python.keras.callbacks.History object at ...>
        >>> clf.fit(iris[features], iris['class']).predict(iris[features])
        0      ...
        1      ...
        2      ...
        3      ...
        4      ...
        ...

    """
    def __init__(self, build_fn, classes, **sk_params):
        KerasEstimator.__init__(self, build_fn, **sk_params)
        self._classes = classes

    def get_params(self, deep=False):
        d = KerasEstimator.get_params(self, deep).copy()
        d['classes'] = self._classes
        return d

    def set_params(self, **sk_params):
        KerasEstimator.set_params(**sk_params)
        self._classes = sk_params['classes']

    def fit(self, X, y, **fit_params):
        """
        Fits the transformer using ``X`` ``y``.

        Arguments:
            X: A :class:`pandas.DataFrame` object.
            y: A :class:`pandas.Series` object whose each entry is in classes.

        Returns:

            ``self``
        """
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
        self._history = self._est.fit(uX, uy.values, **fit_params)
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

    @property
    def classes(self):
        """
        Returns:

            The values of classes classified.
        """
        return self._classes


class KerasRegressor(KerasEstimator, base.RegressorMixin):
    """
    A tensorflow/keras regressor.

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.tensorflow.contrib.keras.wrappers.scikit_learn import KerasRegressor as PdKerasRegressor
        >>> import tensorflow
        ...
        ...
        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])
        ...
        >>> def build_regressor_nn():
        ...     model = tensorflow.contrib.keras.models.Sequential()
        ...     model.add(
        ...         tensorflow.contrib.keras.layers.Dense(20, input_dim=4, activation='relu'))
        ...     model.add(
        ...         tensorflow.contrib.keras.layers.Dense(1))
        ...     model.compile(loss='mean_squared_error', optimizer='adagrad')
        ...     return model
        ...
        >>> prd = PdKerasRegressor(
        ...     build_fn=build_regressor_nn,
        ...     verbose=0)
        >>> prd.fit(iris[features], iris['class'])
        Adapter[KerasRegressor](...build_fn=<function build_regressor_nn at ...)
        >>> prd.history_
        <tensorflow.contrib.keras.python.keras.callbacks.History object at ...>
        >>> prd.fit(iris[features], iris['class']).predict(iris[features])
        0      ...
        1      ...
        2      ...
        3      ...
        4      ...

    """

    def __init__(self, build_fn, **sk_params):
        KerasEstimator.__init__(self, build_fn, **sk_params)

    def fit(self, X, y, **fit_params):
        """
        Fits the transformer using ``X`` ``y``.

        Arguments:
            X: A :class:`pandas.DataFrame` object.
            y: A :class:`pandas.Series` object.

        Returns:

            ``self``
        """
        verify_x_type(X)
        verify_y_type(y)
        self.x_columns = X.columns
        # Tmp Ami - should go in utils
        if y is not None and not X.index.equals(y.index):
            raise ValueError('Indexes do not match')
        uX = self._x(False, X)
        uy = self._y(y)
        self._est = keras.wrappers.scikit_learn.KerasRegressor(self._build_fn, **self._sk_params)
        self._history = self._est.fit(uX, uy, **fit_params)
        return self

    def predict(self, X):
        verify_x_type(X)
        uX = self._x(False, X)
        res = self._est.predict(uX)
        return self._process_wrapped_call_res(False, X, res)

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


for est, adp in [
        (keras.wrappers.scikit_learn.KerasClassifier, KerasClassifier),
        (keras.wrappers.scikit_learn.KerasRegressor, KerasRegressor)]:
    for wrap in wrapped_fn_names:
        if wrap in ['fit']:
            continue
        if not hasattr(est, wrap) and hasattr(adp, wrap):
            delattr(adp, wrap)
        if not  hasattr(adp, wrap):
            continue
        elif six.callable(getattr(adp, wrap)):
            try:
                update_method_wrapper(adp, est, wrap)
            except AttributeError:
                pass


__all__ = ['KerasClassifier', 'KerasRegressor']



