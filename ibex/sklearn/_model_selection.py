from __future__ import absolute_import


import numpy as np
import pandas as pd

import ibex


def _make_cv_params(estimator, orig_X, orig_y):
    class _Est(type(estimator)):
        def fit(self, X, y=None, **fit_params):
            inds = X[:, 0]
            X, y = orig_X.ix[inds], orig_y.ix[inds]
            return super(_Est, self).fit(X, y, **fit_params)

        def predict(self, X):
            inds = X[:, 0]
            X = orig_X.ix[inds]
            return super(_Est, self).predict(X).values

    n = len(orig_X)
    X_ = np.arange(n).reshape((n, 1))
    y_ = None if orig_y is None else np.arange(n)

    return _Est(), X_, y_


def cross_val_predict(estimator, X, y=None, groups=None, cv=None, n_jobs=1,
                      verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                      method='predict'):
    """
    Generate cross-validated estimates for each input data point
    Read more in the :ref:`User Guide <cross_validation>`.

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

    est, X_, y_ = _make_cv_params(estimator, X, y)

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


def _update_module():
    import ibex
    from ibex.sklearn import model_selection as pd_model_selection

    pd_model_selection.cross_val_predict = cross_val_predict

