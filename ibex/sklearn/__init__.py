"""
Wrappers for :mod:`sklearn`.


This module loads corresponding modules in ``sklearn`` on demand, just
as ``sklearn`` does. Its contents depend on those of the ``sklearn``
in your system when it is loaded.

Example:

    >>> import sklearn
    >>> import ibex

    :mod:`sklearn.linear_model` is part of ``sklearn``,
    therefore :mod:`ibex.sklearn` will have a counterpart.

    >>> 'linear_model' in sklearn.__all__
    True
    >>> 'linear_model' in ibex.sklearn.__all__
    True
    >>> from ibex.sklearn import linear_model # doctest: +SKIP

    ``foo`` is not part of ``sklearn``,
    therefore :mod:`ibex.sklearn` will not have a counterpart.

    >>> 'foo' in sklearn.__all__
    False
    >>> 'foo' in ibex.sklearn.__all__
    False
    >>> try:
    ...     from ibex.sklearn import foo
    ... except ImportError:
    ...     print('caught')
    caught

"""


from __future__ import absolute_import


import sys
import imp
import string

import six
import sklearn
import numpy as np
import pandas as pd
from sklearn import base

from ._utils import get_matching_estimators


__all__ = sklearn.__all__


_sklearn_ver = int(sklearn.__version__.split('.')[1])


_X = pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]})
_y = pd.Series([1, 0, 1])


def coef_(self, base_ret):
    if len(base_ret.shape) == 1:
        return pd.Series(base_ret, index=self.x_columns)

    if len(base_ret.shape) == 2:
        index = self.y_columns if self.y_columns is not None else self.classes_
        return pd.DataFrame(base_ret, index=index, columns=self.x_columns)

    raise RuntimeError()


def intercept_(self, base_ret):
    # Tmp Ami - replace next by is_nummeric or is_scalar
    if isinstance(base_ret, (type(1), type(1.), type(1 + 1j))):
        return base_ret

    if len(base_ret.shape) == 1:
        return pd.Series(base_ret)

    raise RuntimeError()


def _get_estimator_extras(orig, est):
    orig_attrs = set(dir(est()))
    try:
        final_attrs = set(dir(est().fit(_X, _y)))
    except TypeError:
        final_attrs = set(dir(est().fit(_X)))
    delta_attrs = final_attrs.difference(orig_attrs)
    delta_attrs = [a for a in delta_attrs if not a.startswith('_')]
    delta_attrs = [a for a in delta_attrs if not a.startswith('n_')]
    return {
        'attribs': []
    }



_code = string.Template('''
"""
Auto-generated :mod:`ibex.sklearn` wrapper for :mod:`sklearn.$mod_name`.
"""


from __future__ import absolute_import as _absolute_import

import traceback as _traceback
import inspect as _inspect

import sklearn as _sklearn
from sklearn import $mod_name as _orig
_orig_all = _orig.__all__
from sklearn import base

import ibex


for name in _orig_all:
    if name.startswith('_'):
        continue
    est = getattr(_orig, name)
    try:
        if not _inspect.isclass(est) or not issubclass(est, base.BaseEstimator):
            globals()[name] = est
            continue
    except TypeError as e:
        globals()[name] = est
        continue
    try:
        extras = ibex.sklearn._get_estimator_extras(_orig, est)
        extra_attribs = extras['attribs']
    except:
        # _traceback.print_exc()
        extra_attribs = []
    try:
        globals()[name] = ibex.frame_ex(
            getattr(_orig, est.__name__),
            extra_attribs=extra_attribs)
    except TypeError as e:
        # _traceback.print_exc()
        globals()[name] = est
''')


class _NewModuleLoader(object):
    """
    Load the requested module via standard import, or create a new module if
    not exist.
    """

    def load_module(self, full_name):
        orig = full_name.split('.')[-1]

        mod = sys.modules.setdefault(full_name, imp.new_module(full_name))
        mod.__file__ = ''
        mod.__name__ = full_name
        mod.__path__ = ''
        mod.__loader__ = self
        mod.__package__ = '.'.join(full_name.split('.')[:-1])

        code = _code.substitute({'mod_name': orig})

        six.exec_(code, mod.__dict__)

        if orig in ['cross_validation', 'model_selection']:
            from ._cross_val_predict import update_module as _cross_val_predict_update_module
            _cross_val_predict_update_module(mod)
        if orig == 'cluster':
            from ._cluster import update_module as _cluster_update_module
            _cluster_update_module(mod)
        if orig == 'decomposition':
            from ._decomposition import update_module as _decomposition_update_module
            _decomposition_update_module(mod)
        if orig == 'ensemble':
            from ._ensemble import update_module as _ensemble_update_module
            _ensemble_update_module(mod)
        if orig == 'feature_selection':
            from ._feature_selection import update_module as _feature_selection_update_module
            _feature_selection_update_module(mod)
        if orig == 'linear_model':
            from ._linear_model import update_module as _linear_model_update_module
            _linear_model_update_module(mod)
        if orig == 'pipeline':
            from ._pipeline import update_module as _pipeline_update_module
            _pipeline_update_module(mod)
        if orig == 'preprocessing':
            from ._preprocessing import update_module as _preprocessing_update_module
            _preprocessing_update_module(mod)
        from ._predict_star_proba import update_module as _predict_star_proba_update_module
        _predict_star_proba_update_module(mod)

        return mod


class _ModuleFinder(object):
    def install(self):
        sys.meta_path[:] = [x for x in sys.meta_path if self != x] + [self]

    def find_module(self, full_name, _=None):
        if not full_name.startswith('ibex.sklearn.'):
            return

        if full_name.split('.')[-1].startswith('_'):
            return

        return _NewModuleLoader()


loader = _ModuleFinder()
loader.install()

