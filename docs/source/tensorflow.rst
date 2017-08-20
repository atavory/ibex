.. _tensorflow:

``tensorflow``
===========

.. versionadded:: 1.2


Ibex wraps :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`. This is straightforward (it has only two scikit-learn type estimators: :class:`tensorflow.contrib.keras.wrappers.scikit_learn.KerasClassifier` and :class:`tensorflow.contrib.keras.wrappers.scikit_learn.KerasRegressor`). The :mod:`ibex.tensorflow` section in the API describes the (straightforward) Ibex corresponding classes.


.. note::

    Ibex does not *require* the installation of :mod:`tensorflow`. If ``tensorflow`` is not installed on the system, though, then this module will not be available.


.. tip::

    Ibex does not modify the code of ``tensorflow`` in any way. It is absolutely possibly to ``import`` and use both ``tensorflow`` and ``ibex.tensorflow`` simultaneously.



