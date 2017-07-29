from __future__ import absolute_import


def _update_module():
    import ibex
    from ibex.sklearn import preprocessing as _pd_preprocessing

    _pd_preprocessing.FunctionTransformer = ibex._FunctionTransformer
