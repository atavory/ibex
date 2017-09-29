from __future__ import absolute_import


import collections
import functools
import itertools
import os
import threading

import pandas as pd
from sklearn import base
from sklearn import pipeline
from sklearn.externals import joblib
try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    from sklearn.utils.validation import NotFittedError # Older Versions

from ._utils import verify_x_type, verify_y_type


__all__ = []


# Tmp Ami - uts, docs
def _make_pipeline_steps(objs):
    names = [type(o).__name__.lower() for o in objs]
    name_counts = collections.Counter(names)
    name_inds = name_counts.copy()
    unique_names = []
    for name in names:
        if name_counts[name] > 1:
            unique_names.append(name + '_' + str(name_counts[name] - name_inds[name]))
            name_inds[name] -= 1
        else:
            unique_names.append(name)

    return list(zip(unique_names, objs))


class FrameMixin(object):
    """
    A base class for steps taking pandas entities, not numpy entities.

    Subclass this step to indicate that a step takes pandas entities.

    Example:

        This is a simple, illustrative "identity" transformer,
        which simply relays its input.

        >>> import pandas as pd
        >>> from sklearn import base
        >>> import ibex
        >>>
        >>> class Id(
        ...            base.BaseEstimator, # (1)
        ...            base.TransformerMixin, # (2)
        ...            ibex.FrameMixin): # (3)
        ...
        ...     def fit(self, X, y=None):
        ...         self.x_columns = X.columns # (4)
        ...         if y is not None and isinstance(y, pd.DataFrame):
        ...             self.y_columns = y.columns
        ...         return self
        ...
        ...     def transform(self, X, *args, **kwargs):
        ...         return X[self.x_columns] # (5)

        Note the following general points:

        1. We subclass :class:`sklearn.base.BaseEstimator`, as this is an estimator.

        2. We subclass :class:`sklearn.base.TransformerMixin`, as, in this case, this is specifically a transformer.

        3. We subclass :class:`ibex.FrameMixin`, as this estimator deals with ``pandas`` entities.

        4. In ``fit``, we make sure to set :py:attr:`ibex.FrameMixin.x_columns`;, and, if relevant,
        :py:attr:`ibex.FrameMixin.y_columns` (if ``y`` is a :class:`pandas.DataFrame`); this will ensure that the
        transformer will "remember" the columns it should see in further calls.

        5. In ``transform``, we first use ``x_columns``. This will verify the columns of ``X``, and also reorder
        them according to the original order seen in ``fit`` (if needed).

        Suppose we define two :class:`pandas.DataFrame` objects, ``X_1`` and ``X_2``, with different columns:

        >>> import pandas as pd
        >>>
        >>> X_1 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]})
        >>> X_2 = X_1.rename(columns={'b': 'd'})

        The following ``fit``-``transform`` combination will work:

        >>> Id().fit(X_1).transform(X_1)
        a  b
        0  1  3
        1  2  4
        2  3  5

        The following ``fit``-``transform`` combination will fail:

        >>> try:
        ...     Id().fit(X_1).transform(X_2)
        ... except KeyError:
        ...     print('caught')
        caught

        The following ``transform`` will fail, as the estimator was not fitted:

        >>> try:
        ...     from sklearn.exceptions import NotFittedError
        ... except ImportError:
        ...     from sklearn.utils.validation import NotFittedError # Older Versions
        >>> try:
        ...     Id().transform(X_2)
        ... except NotFittedError:
        ...     print('caught')
        caught

        Steps can be piped into each other:

        >>> (Id() | Id()).fit(X_1).transform(X_1)
        a  b
        0  1  3
        1  2  4
        2  3  5

        Steps can be added:

        >>> (Id() + Id()).fit(X_1).transform(X_1)
        a  b  a  b
        0  1  3  1  3
        1  2  4  2  4
        2  3  5  3  5
    """

    @property
    def x_columns(self):
        """
        The X columns set in the last call to fit.

        Set this property at fit, and call it in other methods:

        """
        try:
            return self.__x_cols
        except AttributeError:
            raise NotFittedError()

    @x_columns.setter
    def x_columns(self, columns):
        self.__x_cols = columns

    @property
    def y_columns(self):
        """
        The y columns set in the last call to fit.

        Set this property at fit, and call it in other methods:

        .. versionadded:: 0.1.2

        """
        try:
            return self.__y_cols
        except AttributeError:
            raise NotFittedError()

    @y_columns.setter
    def y_columns(self, columns):
        self.__y_cols = columns

    def __or__(self, other):
        """
        Pipes the result of this step to other.

        Arguments:
            other: A different step object whose class subclasses this one.

        Returns:
            :py:class:`ibex.sklearn.pipeline.Pipeline`
        """

        if isinstance(self, Pipeline):
            selfs = [e[1] for e in self.steps]
        else:
            selfs = [self]

        if isinstance(other, Pipeline):
            others = [e[1] for e in other.steps]
        else:
            others = [other]

        combined = selfs + others

        return Pipeline(_make_pipeline_steps(combined))

    def __add__(self, other):
        """

        Returns:
            :py:class:`ibex.sklearn.pipeline.FeatureUnion`
        """

        if isinstance(self, FeatureUnion):
            self_features = [e[1] for e in self.transformer_list]
        else:
            self_features = [self]

        if isinstance(other, FeatureUnion):
            other_features = [e[1] for e in other.transformer_list]
        else:
            other_features = [other]

        combined = self_features + other_features

        return FeatureUnion(_make_pipeline_steps(combined))


__all__ += ['FrameMixin']


def _transform(transformer, weight, X, *args, **kwargs):
    res = transformer.transform(X, *args, **kwargs)
    if weight is not None:
        res *= weight
    return res


def _fit_transform(transformer, weight, X, y, *args, **kwargs):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, *args, **kwargs)
    else:
        res = transformer.fit(X, y, *args, **kwargs).transform(X)
    if weight is not None:
        res *= weight
    return res


class FeatureUnion(pipeline.FeatureUnion, base.TransformerMixin, FrameMixin):
    """
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

        transformer_weights: dict, optional
            Multiplicative weights for features per transformer.
            Keys are transformer names, values the weights.

    Example:

        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [10, -3, 4]})

        >>> from ibex.sklearn import preprocessing as pd_preprocessing
        >>> from ibex.sklearn import pipeline as pd_pipeline

        >>> trn = pd_pipeline.FeatureUnion([
        ...     ('std', pd_preprocessing.StandardScaler()),
        ...     ('asb', pd_preprocessing.MaxAbsScaler())])
        >>> trn.fit_transform(X, *args, **kwargs)
                a         b         a    b
        0 -1.224745  1.192166  0.333333  1.0
        1  0.000000 -1.254912  0.666667 -0.3
        2  1.224745  0.062746  1.000000  0.4

        >>> from ibex import trans
        >>>
        >>> trn = pd_preprocessing.StandardScaler() + pd_preprocessing.MaxAbsScaler()
        >>> trn.fit_transform(X, *args, **kwargs)
                a         b         a    b
        0 -1.224745  1.192166  0.333333  1.0
        1  0.000000 -1.254912  0.666667 -0.3
        2  1.224745  0.062746  1.000000  0.4

        >>> trn = trans(pd_preprocessing.StandardScaler(), out_cols=['std_scale_a', 'std_scale_b'])
        >>> trn += trans(pd_preprocessing.MaxAbsScaler(), out_cols=['max_scale_a', 'max_scale_b'])
        >>> trn.fit_transform(X, *args, **kwargs)
        std_scale_a  std_scale_b  max_scale_a  max_scale_b
        0    -1.224745     1.192166     0.333333          1.0
        1     0.000000    -1.254912     0.666667         -0.3
        2     1.224745     0.062746     1.000000          0.4
    """
    # Tmp Ami - document as_index
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None, as_index=True):
        pipeline.FeatureUnion.__init__(
            self,
            transformer_list,
            n_jobs,
            transformer_weights)
        FrameMixin.__init__(self)
        self._as_index = as_index

    # Tmp Ami - get docstrings from sklearn.
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits the transformer using ``X`` (and possibly ``y``). Transforms
        ``X`` using the transformers, uses :func:`pandas.concat`
        to horizontally concatenate the results.

        Returns:

            ``self``
        """
        verify_x_type(X)
        verify_y_type(y)

        Xts = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_fit_transform)(trans, weight, X, y, **fit_params) for _, trans, weight in self._iter())
        return self.__concat(Xts)

    def transform(self, X, *args, **kwargs):
        """
        Transforms ``X`` using the transformers, uses :func:`pandas.concat`
        to horizontally concatenate the results.
        """
        verify_x_type(X)

        Xts = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_transform)(trans, weight, X, *args, **kwargs) for _, trans, weight in self._iter())
        return self.__concat(Xts)

    def _iter(self):
        weights = self.transformer_weights
        if weights is None:
            weights = {}
        return ((name, trans, weights.get(name, None)) for name, trans in self.transformer_list)

    def __concat(self, Xts):
        conc = pd.concat(Xts, axis=1)

        cols = conc.columns
        tups = [(c, ) if not isinstance(c, tuple) else c for c in cols]
        max_tup_len = max(len(t) for t in tups)
        tups = [c + ('', ) * (max_tup_len - len(c)) for c in tups]

        if self._as_index:
            names = [name for (name, _, _) in self._iter()]
            mults = [len(X.columns) for X in Xts]
            tup_heads = [(name, ) * m for (name, m) in zip(names, mults)]
            tup_heads = list(itertools.chain.from_iterable(tup_heads))
            tups = [(h, ) + t for h, t in zip(tup_heads, tups)]

        conc.columns = pd.MultiIndex.from_tuples(tups)

        return conc


FeatureUnion.__name__ = 'FeatureUnion'

_wrapped = [
    'get_feature_names',
    'get_params',
    'set_params',
]

for wrap in _wrapped:
    try:
        functools.update_wrapper(getattr(FeatureUnion, wrap), getattr(pipeline.FeatureUnion, wrap))
    except AttributeError:
        pass


__all__ += ['FeatureUnion']


class Pipeline(pipeline.Pipeline, FrameMixin):
    pass



class InOpChecker(object):
    def __init__(self, f_name):
        flag = '_ibex_adapter_in_op_%s' % hash(os.path.abspath(f_name))
        self.__set = getattr(threading.local(), flag, set())

    def __contains__(self, est):
        return id(est) in self.__set

    def add(self, est):
        self.__set.add(id(est))

    def remove(self, est):
        self.__set.remove(id(est))


__all__ += ['Pipeline']
