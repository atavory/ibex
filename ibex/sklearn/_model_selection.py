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

    return pd.Series(y_hat, index=y.index)


def _update_module():
    import ibex
    from ibex.sklearn import model_selection as pd_model_selection

    pd_model_selection.cross_val_predict = cross_val_predict

