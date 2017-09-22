.. _xgboost:

``xgboost``
===========

.. versionadded:: 1.2


Wrapping up :mod:`xgboost` is straightforward (it has only two scikit-learn type estimators: :class:`xgboost.XGBClassifier` and :class:`xgboost.XGBRegressor`). The :mod:`ibex.xgboost` section in the API describes the (straightforward) Ibex corresponding classes.


.. note::

    Ibex does not *require* the installation of :mod:`xgboost`. If ``xgboost`` is not installed on the system, though, then this module will not be available.


.. tip::

    Ibex does not modify the code of ``xgboost`` in any way. It is absolutely possibly to ``import`` and use both ``xgboost`` and ``ibex.xgboost`` simultaneously.


.. seealso::

    See :mod:`ibex.xgboost` in :ref:`api`.


