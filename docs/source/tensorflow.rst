.. _tensorflow:

``tensorflow``
===========

.. versionadded:: 1.2



General Idea
------------

Ibex wraps :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`, which has only two scikit-learn type estimators: :class:`tensorflow.contrib.keras.wrappers.scikit_learn.KerasClassifier` and :class:`tensorflow.contrib.keras.wrappers.scikit_learn.KerasRegressor`), with a number of changes to make them more 
`scikit-learn compliant <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_ (see :ref:`tensorflow_diffs`).



.. note::

    Ibex does not *require* the installation of :mod:`tensorflow`. If ``tensorflow`` is not installed on the system, though, then this module will not be available.


.. tip::

    Ibex does not modify the code of ``tensorflow`` in any way. It is absolutely possibly to ``import`` and use both ``tensorflow`` and ``ibex.tensorflow`` simultaneously.




.. _tensorflow_diffs:

Differences From :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`
----------------------------------------------------------------------

:mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn` differs from the estimators of :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`, which it wraps, in three ways:

1. In :mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn`, :class:`KerasRegressor` subclasses :class:`sklearn.base.RegressorMixin`, and :class:`KerasRegressor` subclasses :class:`sklearn.base.RegressorMixin`.

    .. uml::
        :caption: Use and absense of subclassing mixins from :mod:`sklearn.base`.

        skinparam monochrome true
        skinparam shadowing false

		namespace tensorflow.contrib.keras.wrappers.scikit_learn {
			class KerasClassifier
			class KerasRegressor
		}

		namespace ibex.tensorflow.contrib.keras.wrappers.scikit_learn {
			.KerasClassifier --> sklearn.base.ClassifierMixin
			.KerasRegressor --> sklearn.base.RegressorMixin
		}

		namespace sklearn.base {
			class ClassifierMixin
			class RegressorMixin
		}

2. In :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`, :class:`KerasClassifier` and :class:`KerasRegressor` each have a ``fit`` method which returns a :class:`tensorflow.keras.History` object. E.g., starting with

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LinearRegression as PdLinearRegression
        >>> import tensorflow
        ...
        ...
        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])
        ...
        >>> def build_regressor_nn():
        ...     model = tensorflow.contrib.keras.models.Sequential()
        ...     model.add(
        ...         tensorflow.contrib.keras.layers.Dense(20, input_dim=4, activation='relu'))
        ...     model.add(
        ...         tensorflow.contrib.keras.layers.Dense(1))
        ...     model.compile(loss='mean_squared_error', optimizer='adagrad')
        ...     return model
        ... 
        >>> prd = tensorflow.contrib.keras.wrappers.scikit_learn.KerasRegressor(
        ...     build_fn=build_regressor_nn, 
        ...     verbose=0)

        Then 

        >>> prd.fit(iris[features].values, iris['class'].values)
        <tensorflow.contrib.keras.python.keras.callbacks.History object at ...>

        This differs from the usual convention in :mod:`sklearn`:

        >>> prd.fit(iris[features].values, iris['class'].values).score(iris[features].values, iris['class'].values)
        <tensorflow.contrib.keras.python.keras.callbacks.History object at ...>
        ...
        AttributeError: 'History' object has no attribute 'score'

        This differs from the usual convention in :mod:`sklearn` of allowing chained methods:

        Conversely, in :mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn`, ``history_`` is an attribute of an estimator following ``fit``:

        >>> import ibex
        >>> prd = ibex.tensorflow.contrib.keras.wrappers.scikit_learn.KerasRegressor(
        ...     build_fn=build_regressor_nn, 
        ...     verbose=0)
        Adapter[KerasRegressor](verbose=0,build_fn=<function build_regressor_nn at ...>)
        >>> prd.fit(iris[features], iris['class'])
        <tensorflow.contrib.keras.python.keras.callbacks.History object at ...>
        >>> prd.history_
        >>> prd.fit(iris[features], iris['class']).score(iris[features], iris['class'])
        ...

