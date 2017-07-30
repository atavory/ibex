Overview
=========

.. uml::

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
