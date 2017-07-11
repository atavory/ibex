import inspect
import types
import functools

import numpy as np
import pandas as pd
import sklearn
from sklearn import pipeline

import _feature_union
import _frame_mixin


def _delegate_getattr_to_step(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return _Adapter(result)
    return wrapper


class _Adapter(_frame_mixin.FrameMixin):
    """
    Adapts a step to a pandas based step.

    The resulting step takes pd entities; if needed, it strips
        them and passes them to the adapted step as numpy entities.

    Arguments:
        a step or pipeline.
    """
    def __init__(self, step):
        class BaseAdded(_Adapter, type(step)):
            pass
        self.__class__ = BaseAdded
        self.__name__ = '_Adapter'

        _frame_mixin.FrameMixin.__init__(self)

        self._step = step


        xy, x, neut = self._get_wrapped_method_names(step)
        print xy, x, neut

        added = []
        for method_name in dir(step):
            if self._try_wrap_method(step, method_name):
                added.append(method_name)

        if isinstance(step, sklearn.pipeline.Pipeline):
            # Tmp Ami - should be recursive
            internal_step = step.steps[-1][1]
            for method_name in dir(internal_step):
                if method_name not in added:
                    continue

    def _get_wrapped_method_names(self, step):
        xy, x, neut = [], [], []
        for method_name in dir(step):
            if self._is_no_wrap_method_name(method_name):
                continue

            try:
                method = getattr(step, method_name)
            except AttributeError:
                continue

            if not callable(method):
                continue

            try:
                args = inspect.getargspec(method).args
            except TypeError:
                continue

            if args[: 3] == ['self', 'X', 'y']:
                xy.append(method_name)
                continue

            if args[: 2] == ['self', 'X']:
                x.append(method_name)
                continue

            neut.append(method_name)

        return set(xy), set(x), set(neut)

    def __getattr__(self, name):
        result = getattr(self._step, name)
        if callable(result):
            result = _delegate_getattr_to_step(result)
        return result

    def _x(self, x):
        return x if _frame_mixin.FrameMixin.is_subclass(self._step) else x.as_matrix()

    def _y(self, y):
        if y is None:
            return None
        return y if _frame_mixin.FrameMixin.is_subclass(self._step) else y.values

    # Tmp Ami - check if next two are needed
    def get_params(self, deep=True):
        """
        See sklearn.base.BaseEstimator.get_params
        """
        return self._step.get_params(deep)

    def set_params(self, *params):
        """
        See sklearn.base.BaseEstimator.set_params
        """
        return self._step.set_params(*params)

    def _try_wrap_method(self, step, method_name):
        if self._is_no_wrap_method_name(method_name):
            return False

        try:
            method = getattr(step, method_name)
        except AttributeError:
            return False

        if not callable(method):
            return False

        try:
            args = inspect.getargspec(method).args
        except TypeError:
            return False

        if args[: 3] == ['self', 'X', 'y']:
            self.__setattr__(method_name, types.MethodType(self._xy_wrapper(method), step))
            return True

        if args[: 2] == ['self', 'X']:
            self.__setattr__(method_name, types.MethodType(self._x_wrapper(method), step))
            return True

        # Tmp Ami - add here
        return True

    def _is_no_wrap_method_name(self, method_name):
        return method_name.startswith('_')

    def _xy_wrapper(self, method):
        @functools.wraps(method)
        def xy_wrapped(step, X, *args, **kwargs):
            print 'xy_wrapped', method
            self._set_x(X)
            param_X = self._x(X)
            if len(args) > 0:
                y = self._y(args[0])
                args = args[1: ]
                ret = method(param_X, y, *args, **kwargs)
            else:
                ret = method(param_X, *args, **kwargs)

            return self._process_wrapped_call_res(step, X, ret)
        return xy_wrapped

    def _x_wrapper(self, method):
        @functools.wraps(method)
        def x_wrapped(step, X, *args, **kwargs):
            print 'x_wrapped'
            self._set_x(X)
            ret = method(self._x(X), *args, **kwargs)

            return self._process_wrapped_call_res(step, X, ret)
        return x_wrapped

    def _process_wrapped_call_res(self, step, X, ret):
        if isinstance(ret, np.ndarray):
            if len(ret.shape) == 1:
                return pd.Series(ret, index=X.index)
            if len(ret.shape) == 2:
                return pd.DataFrame(ret, index=X.index, columns=X.columns)

        if ret == step:
            return self

        return ret


def frame(step):
    return _Adapter(step)
