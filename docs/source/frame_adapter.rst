.. _adapting:

Low-Level Interface
===================

This chapter describes a low level-class, :class:`ibex.FrameMixin`, indicating 

and :func:`ibex.frame`.

.. tip::

    This chapter describes interfaces required for writing a :mod:`pandas`-aware estimator from scratch, or for adapting an estimator to be ``pandas``-aware. As all of :mod:`sklearn` is wrapped by Ibex, this can be skipped if you're not planning on doing either. 

`adapter <https://en.wikipedia.org/wiki/Adapter_pattern>`_
