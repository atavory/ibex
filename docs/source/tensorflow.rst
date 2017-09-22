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


.. seealso::

    * See :mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn` in :ref:`api`.

    * See `Tensorflow/Keras Classification In The Iris Dataset <https://github.com/atavory/ibex/blob/master/examples/iris_tensorflow.ipynb>`_ in :ref:`examples`.


.. _tensorflow_diffs:

Differences From :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`
----------------------------------------------------------------------

:mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn` differs from the estimators of :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`, which it wraps, in three ways:

1. In :mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn`, :class:`KerasClassifier` subclasses :class:`sklearn.base.ClassifierMixin`, and :class:`KerasClassifier` subclasses :class:`sklearn.base.ClassifierMixin`.

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

        >>> prd.fit(iris[features].values, iris['class'].values).predict(iris[features].values)
        Traceback (most recent call last):
        ...
        AttributeError: 'History' object has no attribute 'predict'

        This differs from the usual convention in :mod:`sklearn` of allowing chained methods:

        Conversely, in :mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn`, ``history_`` is an attribute of an estimator following ``fit``:

        >>> import ibex
        >>> prd = ibex.tensorflow.contrib.keras.wrappers.scikit_learn.KerasRegressor(
        ...     build_fn=build_regressor_nn, 
        ...     verbose=0)
        >>> prd.fit(iris[features], iris['class'])
        Adapter[KerasRegressor](verbose=0,build_fn=<function build_regressor_nn at ...)
        >>> prd.history_
        <tensorflow.contrib.keras.python.keras.callbacks.History object at ...>
        >>> prd.fit(iris[features], iris['class']).predict(iris[features])
        0      ...
        1      ...
        2      ...
        3      ...
        4      ...

3. In :mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn`, :class:`KerasClassifier` takes a one-hot encoding of the dependent variable. For example, using the above dataset, if we have

        >>> iris['class'].head()
		0    0.0
		1    0.0
		2    0.0
		3    0.0
		4    0.0
		Name: class, dtype: float64

        then `fit` needs to be used on something like

        >>> pd.get_dummies(iris['class']).head()
			    0.0  1.0  2.0
		0    1    0    0
		1    1    0    0
		2    1    0    0
		3    1    0    0
		4    1    0    0

        which is nonstandard.

        Conversely, in :mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn`, the dependent variable is a :clas:`pandas.Series` (see ee `Tensorflow/Keras Classification In The Iris Dataset <https://github.com/atavory/ibex/blob/master/examples/iris_tensorflow.ipynb>`_ in :ref:`examples`).

