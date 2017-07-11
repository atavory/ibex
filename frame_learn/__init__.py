import operator
import sys
import inspect
import types
import functools

import numpy as np
import pandas as pd
import sklearn
from sklearn import pipeline


_py3 = sys.version_info[0] == 3


def _is_str(s):
    if _py3:
        return isinstance(s, str)
    return isinstance(s, basestring)


__all__ = []


class FrameMixin(object):
    """
    A base class for steps taking pandas entities, not
        numpy entities.

    Subclass this step to indicate that a step takes pandas
        entities.
    """

    def __init__(self):
        self._cols = None

    def _set_x(self, x):
        self._cols = x.columns

    def _tr_x(self, x):
        if set(x.columns) != set(self._cols):
            raise KeyError()
        return x[self._cols]

    @classmethod
    def is_subclass(cls, step):
        """
        Returns:
            Whether a step is a subclass of Stage.

        Arguments:
            step: A Stage or a pipeline.
        """
        if issubclass(type(step), pipeline.Pipeline):
            if not step.steps:
                raise ValueError('Cannot use 0-length pipeline')
            return cls.is_subclass(step.steps[0][1])
        return issubclass(type(step), FrameMixin)

    def __or__(self, other):
        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]

        return pipeline.make_pipeline(self, *others)

    def __ror__(self, other):
        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]
        others += [self]

        return pipeline.make_pipeline(*others)

    def __add__(self, other):
        return FeatureUnion([('0', self), ('1', other)])
        ff
        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]

        return pipeline.make_pipeline(self, *others)

__all__ += ['FrameMixin']


def _to_step(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return _Adapter(result)
    return wrapper


def _xy_wrapper(method, self):
    @functools.wraps(method)
    def xy_wrapped(step, X, *args, **kwargs):
        self._set_x(X)
        param_X = self._x(X)
        if len(args) > 0:
            y = self._y(args[0])
            args = args[1: ]
            ret = method(param_X, y, *args, **kwargs)
        else:
            ret = method(param_X, *args, **kwargs)

        if isinstance(ret, np.ndarray):
            print ret.shape
            if len(ret.shape) == 1:
                return pd.Series(ret, index=X.index)
            if len(ret.shape) == 2:
                print X, X.index, X.columns
                print pd.DataFrame(ret, index=X.index, columns=X.columns)
                return pd.DataFrame(ret, index=X.index, columns=X.columns)

        if ret == step:
            return self

        return ret
    return xy_wrapped


def _x_wrapper(method, self):
    @functools.wraps(method)
    def x_wrapped(step, X, *args, **kwargs):
        print X, args, kwargs
        self._set_x(X)
        ret = method(self._x(X), *args, **kwargs)

        for _ in range(20):
            print 'x', ret

        if isinstance(ret, np.ndarray):
            print ret.shape
            if len(ret.shape) == 1:
                return pd.Series(ret, index=X.index)
            if len(ret.shape) == 2:
                print X, X.index, X.columns
                print pd.DataFrame(ret, index=X.index, columns=X.columns)
                return pd.DataFrame(ret, index=X.index, columns=X.columns)

        if ret == step:
            return self

        return ret
    return x_wrapped


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

        for method_name in dir(step):
            try:
                method = getattr(step, method_name)
            except AttributeError:
                continue

            print 'here', method_name

            # Tmp Ami
            if not callable(method):
                continue

            try:
                args = inspect.getargspec(method).args
            except TypeError:
                continue

            if args[: 3] == ['self', 'X', 'y']:
                print('setting', method_name)
                self.__setattr__(method_name, types.MethodType(_xy_wrapper(method, self), step))
                print self.__dict__
                continue
            elif args[: 2] == ['self', 'X']:
                print('setting', method_name)
                self.__setattr__(method_name, types.MethodType(_x_wrapper(method, self), step))
                print self.__dict__
                continue
            else:
                # Tmp Ami - add here
                continue

    def __getattr__(self, name):
        print('getattr', name)
        result = getattr(self._step, name)
        if callable(result):
            result = _to_step(result)
        return result

    def _x(self, x):
        return x if FrameMixin.is_subclass(self._step) else x.as_matrix()

    def _y(self, y):
        if y is None:
            return None
        return y if FrameMixin.is_subclass(self._step) else y.values

    def _from_x(self, columns, index, xt):
        return xt if FrameMixin.is_subclass(self._step) \
            else pd.DataFrame(xt, columns=columns, index=index)

    def _from_y(self, index, y_hat):
        return y_hat if FrameMixin.is_subclass(self._step)\
            else pd.Series(y_hat, index=index)

    def _from_p(self, index, classes, probs):
        return probs if FrameMixin.is_subclass(self._step)\
            else pd.DataFrame(probs, columns=classes, index=index)

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


def frame(step):
    return _Adapter(step)

__all__ += ['frame']


class FeatureUnion(FrameMixin):
    """
    - Pandas version -
    Concatenates results of multiple transformer objects.
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Arguments:

    transformer_list: list of (string, transformer) tuples.
        List of transformer objects to be applied to the data.
        The first half of each tuple is the name of the transformer.

    n_jobs: int, optional.
        Number of jobs to run in parallel (default 1).

    transformer_weights: dict, optional.
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
    """
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        FrameMixin.__init__(self)

        self._feature_union = pipeline.FeatureUnion(
            transformer_list,
            n_jobs,
            transformer_weights)

    def fit_transform(self, x, y):
        """
        Same signature as any sklearn step.
        """
        xt = self._feature_union.fit_transform(
            x,
            y)

        return pd.DataFrame(xt, index=x.index)

    def fit(self, x, y):
        """
        Same signature as any sklearn step.
        """
        self._feature_union.fit(
            x,
            y)

        return self

    def transform(self, x):
        """
        Same signature as any sklearn step.
        """
        xt = self._feature_union.transform(x)

        return pd.DataFrame(xt, index=x.index)

__all__ += ['FeatureUnion']


class _FunctionTransformer(FrameMixin):
    """
    Applies some step to only some (or one) columns - Pandas version.

    Arguments:
        wh: Something for which df[:, wh]
            is defined, where df is a pandas DataFrame.
        st: A step.

    Example:

        import day_two

        x = np.linspace(-3, 3, 50)
        y = x
        # Apply the step only to the first column of h.
        sm = day_two.sklearn_.preprocessing\
            .FunctionTransformer(
                'moshe',
                day_two.preprocessing.UnivariateSplineSmoother())
        sm.fit(
            pd.DataFrame({'moshe': x, 'koko': x}),
            pd.Series(y))
    """
    def __init__(self, func, pass_y, kw_args, columns):
        FrameMixin.__init__(self)

        self._func, self._pass_y, self._kw_args, self._columns = \
            func, pass_y, kw_args, columns

    def fit(self, x, y, **fit_params):
        # Tmp AmiAdd here call to fit
        return self

    def transform(self, x, y=None):
        # Tmp Ami Add here call to fit
        if self._columns is not None:
# Tmp AMi - refactor next to utility in top of file
            columns = [self._columns] if _is_str(self._columns) else self._columns
            x = x[columns]

        if self._func is None:
            return x

        if isinstance(self._func, dict):
            dfs = []
            for k, v in self._func.items():
                res = pd.DataFrame(v(x))
                columns = [k] if _is_str(k) else k
                res.columns = columns
                dfs.append(res)
            return pd.concat(dfs, axis=1)

        return self._func(x)

    # Tmp Ami - add fit_transform


def apply(func=None, pass_y=False, kw_args=None, columns=None):
    return _FunctionTransformer(func, pass_y, kw_args, columns)

__all__ += ['apply']



