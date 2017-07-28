"""
Wrappers for :mod:`sklearn`.


This module loads corresponding modules in ``sklearn`` by demand, just
as ``sklearn`` does. Its contents depend on those of the ``sklearn``
in your system when it is loaded.

Example:

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
    >>> from ibex.sklearn import foo
    Traceback (most recent call last):
    ...
    ImportError: ...foo...
"""

from __future__ import absolute_import

import sys
import imp

import six
import sklearn


__all__ = sklearn.__all__


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

        place_function_transformer = full_name == 'ibex.sklearn.preprocessing'

        code = '''
"""
Auto-generated :mod:`ibex.sklearn` wrapper for :mod:`sklearn.%s`.
"""

from __future__ import absolute_import

from sklearn import %s as _orig
from sklearn import base

import ibex

for name in dir(_orig):
    # Tmp Ami - here?
    if name == 'FeatureUnion':
        globals()[name] = ibex._FeatureUnion
        continue

    if name == 'Pipeline':
        globals()[name] = ibex._Pipeline
        continue

    est = getattr(_orig, name)
    try:
        if issubclass(est, base.BaseEstimator):
            globals()[name] = ibex.frame(est)
    except TypeError:
        continue

if %d:
    globals()['FunctionTransformer'] = ibex._FunctionTransformer
        ''' % (orig, orig, place_function_transformer)

        six.exec_(code, mod.__dict__)

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


from ibex.sklearn import pipeline as _pd_pipeline

from ._pipeline import make_pipeline as _pd_make_pipeline
_pd_pipeline.make_pipeline = _pd_make_pipeline

from ._pipeline import make_union as _pd_make_union
_pd_pipeline.make_union = _pd_make_union


