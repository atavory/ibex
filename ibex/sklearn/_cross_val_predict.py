from __future__ import absolute_import

try:
    from sklearn.model_selection import cross_val_predict as _orig_cross_val_predict
except ImportError:
    from sklearn.cross_validation import cross_val_predict as _orig_cross_val_predict
import pandas as pd

from .._xy_estimator import make_estimator, make_xy
from .._utils import verify_x_type, verify_y_type
from .__init__ import _sklearn_ver


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

        >>> import pandas as pd
        >>> from ibex.sklearn.linear_model import LinearRegression
        >>> try:
        ...     from ibex.sklearn.model_selection import cross_val_predict
        ... except: # Older sklearn versions
        ...     from ibex.sklearn.cross_validation import cross_val_predict

        >>> df = pd.DataFrame({
        ...         'x': range(100),
        ...         'y': range(100),
        ...     },
        ...     index=['i%d' % i for i in range(100)])

        >>> cross_val_predict(
        ...     LinearRegression(),
        ...     df[['x']],
        ...     df['y'])
        i0     ...
        i1     ...
        i2     ...
        i3     ...
        i4     ...
        i5     ...
        ...

    """
    verify_x_type(X)
    verify_y_type(y)

    est = make_estimator(estimator, X.index)
    X_, y_ = make_xy(X, y)
    if _sklearn_ver > 17:
        y_hat = _orig_cross_val_predict(
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
    else:
        if groups is not None:
            raise ValueError('groups not supported for cross_val_predict in this version of sklearn')
        if method != 'predict':
            raise ValueError('method not supported for cross_val_predict in this version of sklearn')
        y_hat = _orig_cross_val_predict(
            est,
            X_,
            y_,
            cv,
            n_jobs,
            verbose,
            fit_params,
            pre_dispatch)

    if len(y_hat.shape) == 1:
        return pd.Series(y_hat, index=y.index)
    else:
        return pd.DataFrame(y_hat, index=y.index)


def update_module(module):
    attribs = {
        'cross_val_predict': cross_val_predict,
    }

    for attrib in attribs:
        if attrib in dir(module):
            setattr(module, attrib, attribs[attrib])
