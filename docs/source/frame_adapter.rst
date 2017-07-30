.. _adapting:

Adapting Estimators
===================

This chapter describes a low-level function, :func:`ibex.frame`, which transforms `estimators conforming to the sickit-learn protocol <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_ to :mod:`pandas`-aware estimators.

.. tip::

    This chapter describes interfaces required for writing a :mod:`pandas`-aware estimator from scratch, or for adapting an estimator to be ``pandas``-aware. As all of :mod:`sklearn` is wrapped by Ibex, this can be skipped if you're not planning on doing either. 

The ``frame`` function takes an estimator, and returns an `adapter <https://en.wikipedia.org/wiki/Adapter_pattern>`_ of that estimator. This adapter does the 
same thing as the adapted class, except that:

1. It expects to take :class:`pandas.DataFrame` and :class:`pandas.Series` objects, not :class:`numpy.array` objects. It performs verifications on these inputs, 
    and processes the outputs, as described in :ref:`verification_and_processing`.

2. It supports two additional operators: ``|`` for pipelining (see :ref:`pipeline`), and ``+`` for feature unions (see :ref:`feature_union`).

Suppose we start with:

    >>> from sklearn import linear_model 
    >>> from sklearn import preprocessing
    >>> from sklearn import base
    >>> from ibex import frame

We can use ``frame`` to adapt an object:

    >>> prd = frame(linear_model.LinearRegression())
    >>> prd
    Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

We can use ``frame`` to adapt a class:

    >>> PDLinearRegression = frame(linear_model.LinearRegression)
    >>> PDStandardScaler = frame(preprocessing.StandardScaler)

Once we adapt a class, it behaves pretty much like the underlying one. We can construct it in whatever ways it the underlying class supports, for example:

    >>> PDLinearRegression()
    Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    >>> PDLinearRegression(fit_intercept=False)
    Adapter[LinearRegression](copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)

It has the same name as the underlying class:

    >>> PDLinearRegression.__name__
    'LinearRegression'

It subclasses the same mixins of the underlying class:

    >>> isinstance(PDLinearRegression(), base.RegressorMixin)
    True
    >>> isinstance(PDLinearRegression(), base.TransformerMixin)
    False
    >>> isinstance(PDStandardScaler(), base.RegressorMixin)
    False
    >>> isinstance(PDStandardScaler(), base.TransformerMixin)
    True

As can be seen above, though, the string and representation is modified, to signify this is an adapted type:

    >>> PDLinearRegression()
    Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    >>> linear_model.LinearRegression()
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

