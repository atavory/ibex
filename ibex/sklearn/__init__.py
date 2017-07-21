from __future__ import absolute_import

import sys
import imp
import os

import six


class NewModuleLoader(object):
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

        code = """
from __future__ import absolute_import

from sklearn import %s as _orig
from sklearn import base

import ibex

for name in dir(_orig):
    if name == 'FeatureUnion':
        globals()[name] = ibex._FeatureUnion
        continue
    est = getattr(_orig, name)
    try:
        if issubclass(est, base.BaseEstimator):
            globals()[name] = ibex.frame(est)
    except TypeError:
        continue
        """ % orig

        six.exec_(code, mod.__dict__)

        return mod


class _ModuleFinder(object):
    def install(self):
        sys.meta_path[:] = [x for x in sys.meta_path if self != x] + [self]

    def find_module(self, full_name, _=None):
        if full_name.startswith('ibex.sklearn.'):
            return NewModuleLoader()


loader = _ModuleFinder()
loader.install()
