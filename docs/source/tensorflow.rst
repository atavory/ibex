.. _tensorflow:

``tensorflow``
===========

.. versionadded:: 1.2


Ibex wraps :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`. Keras has only two scikit-learn type estimators: :class:`tensorflow.contrib.keras.wrappers.scikit_learn.KerasClassifier` and :class:`tensorflow.contrib.keras.wrappers.scikit_learn.KerasRegressor`), but the mapping is not a simple wrapping as with other libraries, as it seems that these estimators are not quite 
`conforming to the sickit-learn protocol <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_:

#. In :mod:`tensorflow.contrib.keras.wrappers.scikit_learn`, :class:`KerasClassifier` and :class:`KerasRegressor` do not subclass :class:`ClassifierMixin` and :class:`sklearn.base.RegressorMixin` (from :mod:`sklearn.base`), respectively. Conversely, 
in :mod:`ibex.tensorflow.contrib.keras.wrappers.scikit_learn`, they do.

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


.. note::

    Ibex does not *require* the installation of :mod:`tensorflow`. If ``tensorflow`` is not installed on the system, though, then this module will not be available.


.. tip::

    Ibex does not modify the code of ``tensorflow`` in any way. It is absolutely possibly to ``import`` and use both ``tensorflow`` and ``ibex.tensorflow`` simultaneously.



