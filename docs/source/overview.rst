Overview
=========
Goals
-----

Ibex library aims for two (somewhat independent) goals:

The first, primary goal, is providing `pandas <http://pandas.pydata.org/>`_ adapters for `estimators conforming to the sickit-learn protocol <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_, in particular those of `scikit-learn <http://scikit-learn.org/stable/>`_ itself

.. uml::
    :caption: Relation of Ibex to some other packages in the scientific python stack.

    skinparam monochrome true
    skinparam shadowing false

    skinparam package {
        FontColor #777777
        BorderColor lightgrey
    }

    package "Plotting" {
        [seaborn]
        [plotly]
        [matplotlib]
    }

    package "Machine Learning" {
        [sklearn]
        [**ibex**]
    }

    package "Data Structures" {
        [numpy]
        [pandas]
    }

    [sklearn] -> [numpy] : interfaced by
    [matplotlib] -> [numpy] : interfaced by
    [pandas] ..> [numpy] : implemented over
    [seaborn] -> [pandas] : interfaced by
    [plotly] -> [pandas] : interfaced by
    [seaborn] ..-> [matplotlib] : implemented over
    [**ibex**] -> [pandas] : interfaced by
    [**ibex**] ..-> [sklearn] : implemented over


Consider the preceding UML figure. :mod:`numpy` is a (highly efficient) low-level data structure (strictly speaking, it is more of a buffer interface). both :mod:`matplotlib` and :mod:`sklearn` provide a ``numpy`` interface. Subsequently, :mod:`pandas` provided a higher-level interface to ``numpy``, and some plotting libraries, e.g., :mod:`seaborn` provide a ``pandas`` interface to plotting, while being implemented by ``matplotlib``, but . Similarly, the first aim of Ibex is to provide a ``pandas`` interface to machine learning, while being implemented by ``sklearn``.

The second goal is providing easier, and more succinct ways of combining estimators, features, and pipelines.

Motivation
----------

Interface
---------

Ibex has a very small interface. The core library has a single public class and two functions. The rest of the library is a (mainly auto-generated) wrapper for :mod:`sklearn`, with nearly all of the classes and functions having a straightforward correspondence to ``sklearn``.

:py:class:`ibex.FrameMixin` is a mixin class providing both some utilities for :mod:`pandas` support for higher-up classes, as well as pipeline and feature operators. It is described in :ref:`adapting`. :py:func:`ibex.frame` is a function taking an
`estimator conforming to the sickit-learn protocol <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_ (either an object or a class), and returning a ``pandas``-aware estimator (correspondingly, an object or a class). If estimators are already wrapped (which is the case for all of ``sklearn``), it is not necessary to be concerned with these at all.

:py:func:`ibex.trans` is a utility function that creates an estimator applying a regular Python function, or a different estimator, to a :class:`pandas.DataFrame`, optionally specifying the input and output columns. Again, you do not need to use it if you are just planning on using ``sklearn`` estimators.

Ibex (mostly automatically) wraps all of :py:mod:`sklearn` in :py:mod:`ibex.sklearn`. In almost all cases (except those noted explicitly), the wrapping has a direct correspondence with ``sklearn``. 


Documentation Structure
-----------------------

:py:mod:`sklearn.preprocessing`

:py:mod:`ibex.sklearn.preprocessing`

:py:class:`sklearn.preprocessing.FunctionTransformer`

:py:class:`ibex.sklearn.preprocessing.FunctionTransformer`

:py:class:`ibex.sklearn.pipeline.FeatureUnion`
