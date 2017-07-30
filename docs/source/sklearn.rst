.. _sklearn:

``sklearn``
===========

General Idea
------------

Ibex lazily wraps :mod:`sklearn`. When a module starting with ``ibex.sklearn`` is ``import``-ed, Ibex will load the corresponding module from ``sklearn``, wrap the estimator classes using :func:`ibex.frame`, and load the result. For example say we start with

    >>> import sklearn
    >>> import ibex

:mod:`sklearn.linear_model` is part of ``sklearn``,
therefore :mod:`ibex.sklearn` will have a counterpart.

    >>> 'linear_model' in sklearn.__all__
    True
    >>> 'linear_model' in ibex.sklearn.__all__
    True
    >>> from ibex.sklearn import linear_model as pd_linear_model # doctest: +SKIP

``foo`` is not part of ``sklearn``,
therefore :mod:`ibex.sklearn` will not have a counterpart.

    >>> 'foo' in sklearn.__all__
    False
    >>> 'foo' in ibex.sklearn.__all__
    False
    >>> from ibex.sklearn import foo as pd_foo
    Traceback (most recent call last):
    ...
    ImportError: ...foo...

As noted above, Ibex wraps the estimator classes it finds in the module:

    >>> from sklearn import linear_model 
    >>> linear_model.LinearRegression()
    >>> from ibex.sklearn import linear_model as pd_linear_model # doctest: +SKIP
    >>> pd_linear_model.LinearRegression()

.. tip::

    Ibex does not modify the code of ``sklearn`` in any way. It is absolutely possibly to ``import`` and use both ``sklearn`` and ``ibex.sklearn`` simultaneously.


Differences from ``sklearn``
----------------------------

Since :class:`pandas.DataFrame` and :class:`pandas.Series` are not identical to :class:`numpy.array` s (which is the reason to use the former), some changes are made in ``ibex.sklearn`` relative to the corresponding elements in ``sklearn``. :mod:`ibex.sklearn`` in 
:ref:`api` lists the differences.
