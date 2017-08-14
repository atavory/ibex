from __future__ import absolute_import


from sklearn import base
try:
    from sklearn.model_selection import check_cv
except ImportError:
    from sklearn.cross_validation import check_cv
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils.metaestimators import _safe_split
import pandas as pd

from .._base import FrameMixin
from .._function_transformer import FunctionTransformer as PDFunctionTransformer


def _fit_transform(
        estimator,
        X,
        y,
        train,
        test,
        verbose,
        fit_params):

    X_train, y_train = _safe_split(estimator, X, y, train)
    if fit_params is None:
        fit_params = {}
    estimator.fit(X_train, y_train, **fit_params)
    X_test, _ = _safe_split(estimator, X, y, test, train)
    return estimator.transform(X_test)


class Stacker(base.BaseEstimator, base.TransformerMixin, FrameMixin):
    """
    A transformer applying fitting a transformor `estimator` to data in a way
        that will allow a higher-up transformor to build a model utilizing both this
        and other transformors correctly.

    The fit_transform(self, x, y) of this class will create a column matrix, whose
        each row contains the transformion of `estimator` fitted on other rows than this one.
        This allows a higher-level transformor to correctly fit a model on this, and other
        column matrices obtained from other lower-level transformors.

    The fit(self, x, y) and transform(self, x_) methods, will fit `estimator` on all
        of `x`, and transform the output of `x_` (which is either `x` or not) using the fitted
        `estimator`.

    Arguments:
        estimator: A lower-level transformor to stack.

        cv: Function taking `x`, and returning a cross-validation object. In `fit_transform`
            th train and test indices of the object will be iterated over. For each iteration, `estimator` will
            be fitted to the `x` and `y` with rows corresponding to the
            train indices, and the test indices of the output will be obtained
            by transforming on the corresponding indices of `x`.
    """
    def __init__(
            self,
            estimator,
            cv=None,
            n_jobs=1,
            verbose=0,
            pre_dispatch='2*n_jobs'):

        self.set_params(
            estimator=estimator,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch)

    def fit_transform(self, X, y, **fit_params):
        Xt = self._train_transform(X, y, **fit_params)

        self.fit(X, y, **fit_params)

        return Xt

    def fit(self, X, y, **fit_params):
        self.estimator.fit(X, y, **fit_params)

        return self

    def transform(self, X):
        return self._test_transform(X)

    def _train_transform(self, X, y, **fit_params):
        cv = check_cv(self.cv, y, classifier=False)
        n_splits = cv.get_n_splits(X, y, None)
        if self.verbose > 0:
            # Tmp Ami
            print(n_splits)
        cv_iter = list(cv.split(X, y, None))
        out = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                pre_dispatch=self.pre_dispatch
            )(delayed(_fit_transform)(
                clone(self.estimator),
                    X,
                    y,
                    train,
                    test,
                    self.verbose,
                    fit_params=fit_params)
                for train, test in cv_iter)
        return pd.concat(out, axis=0)

    def _test_transform(self, X):
        return self.estimator.transform(X)


def update_module(module):
    setattr(module, 'FunctionTransformer', PDFunctionTransformer)
    setattr(module, 'Stacker', Stacker)
