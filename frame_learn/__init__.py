import operator
import numpy as np
import pandas as pd
import sklearn
from sklearn import pipeline
import sys


_py3 = sys.version_info[0] == 3


def _is_str(s):
    if _py3:
        return isinstance(s, str)
    return isinstance(s, basestring)


__all__ = []


class _Step(object):
    """
    A base class for stages taking pandas entities, not
        numpy entities.

    Subclass this stage to indicate that a stage takes pandas
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
    def is_subclass(cls, stage):
        """
        Returns:
            Whether a stage is a subclass of Stage.

        Arguments:
            stage: A Stage or a pipeline.
        """
        if issubclass(type(stage), pipeline.Pipeline):
            if not stage.steps:
                raise ValueError('Cannot use 0-length pipeline')
            return cls.is_subclass(stage.steps[0][1])
        return issubclass(type(stage), _Step)

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

        return pipeline.make_pipeline(*others, self)

    def __add__(self, other):
        return FeatureUnion([self, other])
        ff
        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]

        return pipeline.make_pipeline(self, *others)

__all__ += ['_Step']


class _Adapter(_Step):
    """
    Adapts a stage to a pandas based stage.

    The resulting stage takes pd entities; if needed, it strips
        them and passes them to the adapted stage as numpy entities.

    Arguments:
        a stage or pipeline.
    """
    def __init__(self, stage):
        _Step.__init__(self)

        self._stage = stage

    def fit(self, x, y=None, **fit_params):
        """
        Same signature as any sklearn stage.
        """
        self._set_x(x)
        self._stage.fit(self._x(x), self._y(y), **fit_params)

        return self

    def fit_transform(self, x, y=None, **fit_params):
        """
        Same signature as any sklearn stage.
        """
        self._set_x(x)
        xt = self._stage.fit_transform(self._x(x), self._y(y), **fit_params)
        return self._from_x(x.columns, x.index, xt)

    def predict(self, x):
        """
        Same signature as any sklearn stage.
        """
        x = self._tr_x(x)
        y_hat = self._stage.predict(self._x(x))
        return self._from_y(x.index, y_hat)

    def predict_proba(self, x):
        """
        Same signature as any sklearn stage.
        """
        x = self._tr_x(x)
        probs = self._stage.predict_proba(self._x(x))
        classes = self._stage.classes_
        return self._from_p(x.index, classes, probs)

    def transform(self, x):
        """
        Same signature as any sklearn stage.
        """
        x = self._tr_x(x)
        xt = self._stage.transform(self._x(x))
        return self._from_x(x.columns, x.index, xt)

    def score(self, x, y):
        x = self._tr_x(x)
        return self._stage.score(self._x(x), self._y(y))

    def _x(self, x):
        return x if _Step.is_subclass(self._stage) else x.as_matrix()

    def _y(self, y):
        if y is None:
            return None
        return y if _Step.is_subclass(self._stage) else y.values

    def _from_x(self, columns, index, xt):
        return xt if _Step.is_subclass(self._stage) \
            else pd.DataFrame(xt, columns=columns, index=index)

    def _from_y(self, index, y_hat):
        return y_hat if _Step.is_subclass(self._stage)\
            else pd.Series(y_hat, index=index)

    def _from_p(self, index, classes, probs):
        return probs if _Step.is_subclass(self._stage)\
            else pd.DataFrame(probs, columns=classes, index=index)

    def get_params(self, deep=True):
        """
        See sklearn.base.BaseEstimator.get_params
        """
        return self._stage.get_params(deep)

    def set_params(self, *params):
        """
        See sklearn.base.BaseEstimator.set_params
        """
        return self._stage.set_params(*params)


def frame(stage):
    return _Adapter(stage)

__all__ += ['frame']


class FeatureUnion(object):
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
        _Step.__init__(self)

        self._feature_union = pipeline.FeatureUnion(
            transformer_list,
            n_jobs,
            transformer_weights)

    def fit_transform(self, x, y):
        """
        Same signature as any sklearn stage.
        """
        xt = self._feature_union.fit_transform(
            x,
            y)

        return pd.DataFrame(xt, index=x.index)

    def fit(self, x, y):
        """
        Same signature as any sklearn stage.
        """
        self._feature_union.fit(
            x,
            y)

        return self

    def transform(self, x):
        """
        Same signature as any sklearn stage.
        """
        xt = self._feature_union.transform(x)

        return pd.DataFrame(xt, index=x.index)

__all__ += ['FeatureUnion']


class _FunctionTransformer(_Step):
    """
    Applies some stage to only some (or one) columns - Pandas version.

    Arguments:
        wh: Something for which df[:, wh]
            is defined, where df is a pandas DataFrame.
        st: A stage.

    Example:

        import day_two

        x = np.linspace(-3, 3, 50)
        y = x
        # Apply the stage only to the first column of h.
        sm = day_two.sklearn_.preprocessing\
            .FunctionTransformer(
                'moshe',
                day_two.preprocessing.UnivariateSplineSmoother())
        sm.fit(
            pd.DataFrame({'moshe': x, 'koko': x}),
            pd.Series(y))
    """
    def __init__(self, func, pass_y, kw_args, columns):
        _Step.__init__(self)

        self._func, self._pass_y, self._kw_args, self._columns = \
            func, pass_y, kw_args, columns

    def fit(self, x, y, **fit_params):
        # Tmp AmiAdd here call to fit
        return self

    def transform(self, x, y=None):
        # Tmp AmiAdd here call to fit
        if self._columns is not None:
            x = x[self._columns]

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

    def predict(self, x):
        uni_x = x[self._wh]
        return self._st.predict(uni_x)

    def predict_proba(self, x):
        uni_x = x[self._wh]
        self.classes_ = self._st._stage.classes_
        return self._st.predict_proba(uni_x)


def apply(func=None, pass_y=False, kw_args=None, columns=None):
    return _FunctionTransformer(func, pass_y, kw_args, columns)

__all__ += ['apply']



