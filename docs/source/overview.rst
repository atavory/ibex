Overview
=========

Ibex library aims for two (somewhat independent) goals:

The first one is providing `pandas <http://pandas.pydata.org/>`_ adapters for `estimators conforming to the sickit-learn protocol <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_, in particular those of `scikit-learn <http://scikit-learn.org/stable/>`_ itself

.. uml::
    :caption: Relation of Ibex to some other packages in the scientific python stack.

    package "Plotting" {
        [seaborn]
        [matplotlib]
    }

    package "Machine Learning" {
        [sklearn]
        [ibex]
    }

    package "Data Structures" {
        [numpy]
        [pandas]
    }

    [sklearn] -> [numpy] : interfaced by
    [matplotlib] -> [numpy] : interfaced by
    [pandas] ..> [numpy] : implemented over
    [seaborn] -> [pandas] : interfaced by
    [seaborn] ..-> [matplotlib] : implemented over
    [ibex] -> [pandas] : interfaced by
    [ibex] ..-> [sklearn] : implemented over


* allowing easier, and more succinct ways of combining estimators, features, and pipelines

:py:class:`ibex.FrameMixin`

:py:func:`ibex.frame`

:py:func:`ibex.trans`

:py:mod:`sklearn`

:py:mod:`ibex.sklearn`

:py:mod:`sklearn.preprocessing`

:py:mod:`ibex.sklearn.preprocessing`

:py:class:`sklearn.preprocessing.FunctionTransformer`

:py:class:`ibex.sklearn.preprocessing.FunctionTransformer`

:py:class:`ibex.sklearn.pipeline.FeatureUnion`
