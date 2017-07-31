from __future__ import absolute_import


import inspect

import six
import numpy as np
import pandas as pd
from sklearn import base

from .._base import FrameMixin


def make_xy_estimator(estimator, orig_X, orig_y=None):
    def get_set_params(est):
        base_attr = getattr(estimator, '__init__')
        if six.PY3:
            args = list(inspect.signature(base_attr).parameters)
        else:
            args = inspect.getargspec(base_attr)[0]
        orig_args = est.get_params()
        args = {arg: orig_args[arg] for arg in args if arg in orig_args}
        return args

    class _Adapter(type(estimator)):
        def fit_transform(self, X, *args):
            return self.__run(super(_Adapter, self).fit_transform, 'fit_transform', X, *args)

        def predict_proba(self, X, *args):
            return self.__run(super(_Adapter, self).predict_proba, 'predict_proba', X, *args)

        def sample_y(self, X, *args):
            return self.__run(super(_Adapter, self).sample_y, 'sample_y', X, *args)

        def score_samples(self, X, *args):
            return self.__run(super(_Adapter, self).score_samples, 'score_samples', X, *args)

        def staged_predict_proba(self, X, *args):
            return self.__run(super(_Adapter, self).staged_predict_proba, 'staged_predict_proba', X, *args)

        def apply(self, X, *args):
            return self.__run(super(_Adapter, self).apply, 'apply', X, *args)

        def bic(self, X, *args):
            return self.__run(super(_Adapter, self).bic, 'bic', X, *args)

        def perplexity(self, X, *args):
            return self.__run(super(_Adapter, self).perplexity, 'perplexity', X, *args)

        def fit(self, X, *args):
            return self.__run(super(_Adapter, self).fit, 'fit', X, *args)

        def decision_function(self, X, *args):
            return self.__run(super(_Adapter, self).decision_function, 'decision_function', X, *args)

        def aic(self, X, *args):
            return self.__run(super(_Adapter, self).aic, 'aic', X, *args)

        def partial_fit(self, X, *args):
            return self.__run(super(_Adapter, self).partial_fit, 'partial_fit', X, *args)

        def predict(self, X, *args):
            return self.__run(super(_Adapter, self).predict, 'predict', X, *args)

        def radius_neighbors(self, X, *args):
            return self.__run(super(_Adapter, self).radius_neighbors, 'radius_neighbors', X, *args)

        def staged_decision_function(self, X, *args):
            return self.__run(super(_Adapter, self).staged_decision_function, 'staged_decision_function', X, *args)

        def staged_predict(self, X, *args):
            return self.__run(super(_Adapter, self).staged_predict, 'staged_predict', X, *args)

        def inverse_transform(self, X, *args):
            return self.__run(super(_Adapter, self).inverse_transform, 'inverse_transform', X, *args)

        def fit_predict(self, X, *args):
            return self.__run(super(_Adapter, self).fit_predict, 'fit_predict', X, *args)

        def kneighbors(self, X, *args):
            return self.__run(super(_Adapter, self).kneighbors, 'kneighbors', X, *args)

        def predict_log_proba(self, X, *args):
            return self.__run(super(_Adapter, self).predict_log_proba, 'predict_log_proba', X, *args)

        def transform(self, X, *args):
            return self.__run(super(_Adapter, self).transform, 'transform', X, *args)

        def score(self, X, *args):
            return self.__run(super(_Adapter, self).score, 'score', X, *args)

        def __run(self, fn, name, X, *args):
            if hasattr(self, '_ibex_in_op'):
                return fn(X, *args)

            inds = X[:, 0]

            base_attr = getattr(type(estimator), name)
            if six.PY3:
                params = list(inspect.signature(base_attr).parameters)
            else:
                params = inspect.getargspec(base_attr)[0]

            # Tmp Ami - write a ut for this; remove todo from docs
            if len(params) > 2 and params[2] == 'y' and len(args) > 0 and args[0] is not None:
                args = list(args)[:]
                args[0] = orig_y.ix[inds]

            self._ibex_in_op = True
            try:
                res = fn(orig_X.ix[inds], *args)
            finally:
                delattr(self, '_ibex_in_op')

            return res

        @property
        def orig_estimator(self):
            est = base.clone(estimator)
            return est.set_params(**get_set_params(self))


    n = len(orig_X)
    X_ = np.arange(n).reshape((n, 1))
    y_ = None if orig_y is None else np.arange(n)

    return _Adapter(**get_set_params(estimator)), X_, y_


def cross_val_predict(
        estimator,
        X,
        y=None,
        groups=None,
        cv=None,
        n_jobs=1,
        verbose=0,
        fit_params=None,
        pre_dispatch='2*n_jobs',
        method='predict'):
    """
    Generate cross-validated estimates for each input data point.

    Arguments:

        estimator: estimator object implementing 'fit' and 'predict'
            The object to use to fit the data.
        X: :class:`pandas.DataFrame`
            The data to fit.
        y: The target variable to try to predict in the case of
            supervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - ``None``, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - An object to be used as a cross-validation generator.
            - An iterable yielding train, test splits.

            For integer/``None`` inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used.
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        n_jobs : integer, optional
            The number of CPUs to use to do the computation. -1 means
            'all CPUs'.
        verbose : integer, optional
            The verbosity level.
        fit_params : dict, optional
            Parameters to pass to the fit method of the estimator.
        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:

                - ``None``, in which case all the jobs are immediately
                    created and spawned. Use this for lightweight and
                    fast-running jobs, to avoid delays due to on-demand
                    spawning of the jobs
                - An int, giving the exact number of total jobs that are
                    spawned
                - A string, giving an expression as a function of n_jobs,
                    as in '2*n_jobs'

        method : string, optional, default: 'predict'
            Invokes the passed method name of the passed estimator.

    Returns:

        :class:`pandas.DataFrame` or :class:`pandas.Series` (depending on ``y``).

    Example:

        >>> from ibex.sklearn import model_selection as pd_model_selection

        >>> df = pd.DataFrame({
        ...         'x': range(100),
        ...         'y': range(100),
        ...     },
        ...     index=['i%d' % i for i in range(100)])

        >>> pd_model_selection.cross_val_predict(
        ...     pd_linear_model.LinearRegression(),
        ...     df[['x']],
        ...     df['y'])
    """

    from sklearn import model_selection

    est, X_, y_ = make_xy_estimator(estimator, X, y)

    y_hat = model_selection.cross_val_predict(
        est,
        X_,
        y_,
        groups,
        cv,
        n_jobs,
        verbose,
        fit_params,
        pre_dispatch,
        method)

    if len(y_hat.shape) == 1:
        return pd.Series(y_hat, index=y.index)
    else:
        return pd.DataFrame(y_hat, index=y.index)


class BaseSearchCV(base.BaseEstimator, FrameMixin):
    def __init__(self, estimator):
        self._estimator = estimator


class GridSearchCV(BaseSearchCV):
    import ibex

    def __init__(
            self,
            estimator,
            param_grid,
            scoring=None,
            fit_params=None,
            n_jobs=1,
            iid=True,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch='2*n_jobs',
            error_score='raise',
            return_train_score=True):

        from sklearn import model_selection

        BaseSearchCV.__init__(self, estimator)

        self._cv = model_selection.GridSearchCV(
            estimator,
            param_grid,
            scoring,
            fit_params,
            n_jobs,
            iid,
            refit,
            cv,
            verbose,
            pre_dispatch,
            error_score,
            return_train_score)

    def fit(self, X, y=None, groups=None):
        params = self._cv.get_params()
        est, X_, y_ = make_xy_estimator(self._estimator, X, y)
        params.update({'estimator': est})
        self._cv.set_params(**params)
        self._cv.fit(X_, y=y_, groups=groups)
        return self

    @property
    def best_estimator_(self):
        return self._cv.best_estimator_.orig_estimator


def update_module():
    import ibex
    from ibex.sklearn import model_selection as pd_model_selection

    pd_model_selection.cross_val_predict = cross_val_predict

