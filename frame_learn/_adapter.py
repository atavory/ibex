import inspect
import types
import functools

import numpy as np
import pandas as pd
import sklearn

from ._frame_mixin import FrameMixin


def _delegate_getattr_to_step(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return _Adapter(result)
    return wrapper


class _Adapter(FrameMixin):
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

        FrameMixin.__init__(self)

        self._step = step

        xy, x, neut = self._get_wrapped_method_names(step)

        if isinstance(step, sklearn.pipeline.Pipeline):
            step_xy, step_x, _ = \
                self._get_wrapped_method_names(step.steps[-1][1])

            xy |= step_xy.intersection(neut)
            neut -= step_xy.intersection(neut)

            x |= step_x.intersection(neut)
            neut -= step_x.intersection(neut)

        for method_name in xy:
            method = getattr(step, method_name)
            self.__setattr__(
                method_name,
                types.MethodType(self._xy_wrapper(method), step))

        for method_name in x:
            method = getattr(step, method_name)
            self.__setattr__(
                method_name,
                types.MethodType(self._x_wrapper(method), step))

        for method_name in neut:
            method = getattr(step, method_name)
            self.__setattr__(
                method_name,
                types.MethodType(self._neut_wrapper(method), step))

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
        return x if FrameMixin.is_subclass(self._step) else x.as_matrix()

    def _y(self, y):
        if y is None:
            return None
        return y if FrameMixin.is_subclass(self._step) else y.values

    def _is_no_wrap_method_name(self, method_name):
        return method_name.startswith('_')

    def _xy_wrapper(self, method):
        @functools.wraps(method)
        def xy_wrapped(step, X, *args, **kwargs):
            self.set_x(X)
            param_X = self._x(X)
            if len(args) > 0:
                y = self._y(args[0])
                args = args[1:]
                ret = method(param_X, y, *args, **kwargs)
            else:
                ret = method(param_X, *args, **kwargs)

            return self._process_wrapped_call_res(step, X, ret)
        return xy_wrapped

    def _x_wrapper(self, method):
        @functools.wraps(method)
        def x_wrapped(step, X, *args, **kwargs):
            self.set_x(X)
            ret = method(self._x(X), *args, **kwargs)

            return self._process_wrapped_call_res(step, X, ret)
        return x_wrapped

    def _neut_wrapper(self, method):
        @functools.wraps(method)
        def neut_wrapped(_, *args, **kwargs):
            return method(*args, **kwargs)
        return neut_wrapped

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
