"""
Wrappers for :mod:`tensorflow.contrib.keras.wrappers.sklearn`.

.. versionadded:: 1.2

"""


from __future__ import absolute_import

from sklearn import base
from tensorflow.contrib import keras

from ......_adapter import frame_ex


def __str__(self, _):
    return self.__repr__()


def __repr__(self, _):
    cls_name = 'KerasClassifier' if isinstance(self, base.ClassifierMixin) else 'KerasRegressor'
    return 'Adapter[' + cls_name + '](' + str(self.get_params()) + ')'


KerasClassifier = frame_ex(
    keras.wrappers.scikit_learn.KerasClassifier,
    extra_methods=[__str__, __repr__])
KerasRegressor = frame_ex(
    keras.wrappers.scikit_learn.KerasRegressor,
    extra_methods=[__str__, __repr__])


__all__ = ['KerasClassifier', 'KerasRegressor']

