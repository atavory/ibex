"""
Wrappers for :mod:`tensorflow.contrib.keras.wrappers.sklearn`.

.. versionadded:: 1.2

"""


from __future__ import absolute_import

import functools

from sklearn import base
from tensorflow.contrib import keras

from ......_adapter import frame_ex


def __str__(self, _):
    return self.__repr__()


def __repr__(self, _):
    cls_name = 'KerasClassifier' if isinstance(self, base.ClassifierMixin) else 'KerasRegressor'
    return 'Adapter[' + cls_name + '](' + str(self.get_params()) + ')'


def _make_fit(fn):
    @functools.wraps(fn)
    def wrapped(self, X, y):
        self.history_ = fn(self, X, y)
        return self
    return wrapped


def _correct(cls):
    cls.fit = _make_fit(cls.fit)


KerasClassifier = frame_ex(
    keras.wrappers.scikit_learn.KerasClassifier,
    extra_methods=[__str__, __repr__],
    post_op=_correct)
KerasRegressor = frame_ex(
    keras.wrappers.scikit_learn.KerasRegressor,
    extra_methods=[__str__, __repr__],
    post_op=_correct)


__all__ = ['KerasClassifier', 'KerasRegressor']

