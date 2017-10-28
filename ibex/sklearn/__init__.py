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


__all__ = sklearn.__all__


_code = string.Template('''
"""
Auto-generated :mod:`ibex.sklearn` wrapper for :mod:`sklearn.$mod_name`.
"""


from __future__ import absolute_import


import inspect

import sklearn
try:
    from sklearn import $mod_name as _orig
    _orig_all = _orig.__all__
except ImportError:
    _ver = int(sklearn.__version__.split('.')[1])
    if _ver < 18 and $mod_name in ['model_selection']:
        _orig_all = []
    else:
        raise
from sklearn import base

import ibex


for name in _orig_all:
    if name.startswith('_'):
        continue
    est = getattr(_orig, name)
    try:
        if inspect.isclass(est) and issubclass(est, base.BaseEstimator):
            globals()[name] = ibex.frame(est)
        else:
            globals()[name] = est
    except TypeError as e:
        pass''')


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

