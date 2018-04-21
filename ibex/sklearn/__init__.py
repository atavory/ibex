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
import re
import traceback

import six
import sklearn
import pandas as pd

from .._base import Pipeline as PdPipeline
from .._base import FeatureUnion as PdFeatureUnion
from .._base import _get_fit_doc


__all__ = sklearn.__all__


_int_re = re.compile(r'(\d+)')
_sklearn_ver = int(
    _int_re.search(sklearn.__version__.split('.')[1]).groups()[0])


def _replace(orig, name):
    if orig in ['cross_validation', 'model_selection']:
        from . import _cross_val_predict
        if name == 'cross_val_predict':
            return _cross_val_predict.cross_val_predict

    if orig == 'pipeline':
        if name ==  'Pipeline':
            return PdPipeline
        if name == 'FeatureUnion':
            return PdFeatureUnion

    if orig == 'preprocessing':
        if name == 'FunctionTransformer':
            from .._function_transformer import FunctionTransformer as PdFunctionTransformer
            return PdFunctionTransformer


def _add(orig):
    if orig == 'pipeline':
        from . import _pipeline
        return {
            'make_union': _pipeline.make_union,
            'make_pipeline': _pipeline.make_pipeline,
        }

    return {}


_X = pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]})
_y = pd.Series([1, 0, 1])


def _get_init_params(orig, name):
    kwargs = {}

    if orig == 'feature_selection' and name == 'SelectKBest':
        kwargs = {'k': 1}

    if orig == 'cluster' and name == 'KMeans':
        kwargs = {'n_clusters': 3, 'random_state': 1}

    if orig == 'svm' and name in ['NuSVC', 'NuSVR', 'SVC', 'SVR']:
        kwargs = {'kernel': 'linear'}

    return kwargs


def _get_estimator_extras(orig, name, est):
    kwargs = _get_init_params(orig, name)

    try:
        est(**kwargs).fit(_X, _X)
        has_dataframe_y = True
    except:
        has_dataframe_y = False

    orig_attrs = set(dir(est(**kwargs)))
    try:
        final_attrs = set(dir(est(**kwargs).fit(_X, _y)))
    except TypeError:
        try:
            final_attrs = set(dir(est(**kwargs).fit(_X)))
        except ValueError:
            final_attrs = set(dir(est(**kwargs).fit(_y)))
    final_attrs = final_attrs.union(orig_attrs)
    final_attrs = [a for a in final_attrs if not a.startswith('_')]
    final_attrs = [a for a in final_attrs if not a.startswith('n_')]
    attrs = {}
    methods = {}

    is_regressor = issubclass(est, sklearn.base.RegressorMixin)
    is_classifier = issubclass(est, sklearn.base.ClassifierMixin)
    is_clusterer = issubclass(est, sklearn.base.ClusterMixin)
    is_transformer = issubclass(est, sklearn.base.TransformerMixin)

    # Tmp Ami - basestr
    value_form = lambda v: ("'%s'" % str(v)) if isinstance(v, str) else v
    kw_args_str = ', '.join('%s=%s' % (k, value_form(v)) for k, v in kwargs.items())

    methods['fit'] = (None,
        _get_fit_doc(
            orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))

    if orig == 'decomposition':
        from . import _decomposition
        methods['transform'] = (_decomposition.transform,
            _decomposition.get_transform_doc(
                orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))
        methods['fit_transform'] = (_decomposition.fit_transform,
            _decomposition.get_fit_transform_doc(
                orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))
        attrs['components_'] = (_decomposition.components_,
            _decomposition.get_components_doc(
                orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))

    if orig == 'feature_selection' and hasattr(est, 'get_support'):
        from . import _feature_selection
        methods['transform'] = (_feature_selection.transform,
            _feature_selection.get_transform_doc(
                orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))
        methods['fit_transform'] = (_feature_selection.fit_transform,
            _feature_selection.get_fit_transform_doc(
                orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))

    if orig == 'preprocessing':
        if name == 'PolynomialFeatures':
            from . import _preprocessing
            methods['transform'] = (_preprocessing.polynomial_feautures_transform_imp,
                _preprocessing.polynomial_feautures_transform_imp_doc)
            methods['fit_transform'] = (_preprocessing.polynomial_feautures_transform_imp,
                _preprocessing.polynomial_feautures_transform_imp_doc)

    if is_clusterer:
        from . import _cluster
        methods['transform'] = (_cluster.transform,
            _cluster.get_transform_doc(
                orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))
        methods['fit_transform'] = (_cluster.fit_transform,
            _cluster.get_fit_transform_doc(
                orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))

    if 'intercept_' in final_attrs:
        if is_classifier:
            from . import _classification_coef_intercept
            attrs['intercept_'] = (_classification_coef_intercept.intercept_,
                _classification_coef_intercept.get_intercept_doc(
                    orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))
        else:
            from . import _regression_coef_intercept
            attrs['intercept_'] = (_regression_coef_intercept.intercept_,
                _regression_coef_intercept.get_intercept_doc(
                    orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))
    if 'coef_' in final_attrs:
        if is_classifier:
            from . import _classification_coef_intercept
            attrs['coef_'] = (_classification_coef_intercept.coef_,
                _classification_coef_intercept.get_coef_doc(
                    orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))
        else:
            from . import _regression_coef_intercept
            attrs['coef_'] = (_regression_coef_intercept.coef_,
                _regression_coef_intercept.get_coef_doc(
                    orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))
    if 'feature_importances_' in final_attrs:
        from . import _feature_importances
        attrs['feature_importances_'] = (_feature_importances.feature_importances_,
            _feature_importances.get_feature_importances_docs(
                orig, name, est, kw_args_str, is_regressor, is_classifier, is_transformer, is_clusterer, has_dataframe_y))

    return {
        'attrs': attrs,
        'methods': methods,
    }


_code = string.Template('''
"""
Auto-generated :mod:`ibex.sklearn` wrapper for :mod:`sklearn.$mod_name`.
"""


from __future__ import absolute_import as _absolute_import

import traceback as _traceback
import inspect as _inspect
import sys as _sys
import os as _os

import sklearn as _sklearn
from sklearn import $mod_name as _orig
_orig_all = _orig.__all__
from sklearn import base

import ibex


for name in _orig_all:
    if name.startswith('_'):
        continue
    est = getattr(_orig, name)

    if ibex.sklearn._replace('$mod_name', name):
        globals()[name] = ibex.sklearn._replace('$mod_name', name)
        continue

    try:
        if not _inspect.isclass(est) or not issubclass(est, base.BaseEstimator):
            globals()[name] = est
            continue
    except TypeError as e:
        globals()[name] = est
        continue

    try:
        extras = ibex.sklearn._get_estimator_extras('$mod_name', name, est)
        extra_attribs = extras['attrs']
        extra_methods = extras['methods']
    except:
        if _os.getenv('IBEX_TEST_LEVEL'):
            _traceback.print_exc()
        _sys.stderr.write(str(est))
        extra_attribs = {}
        extra_methods = {}

    try:
        globals()[name] = ibex.frame_ex(
            getattr(_orig, est.__name__),
            extra_attribs=extra_attribs,
            extra_methods=extra_methods)
    except TypeError as e:
        if _os.getenv('IBEX_TEST_LEVEL'):
            _traceback.print_exc()
        globals()[name] = est

_add = ibex.sklearn._add('$mod_name')
for name in _add:
    globals()[name] = _add[name]
''')


class _NewModuleLoader(object):
    """
    Load the requested module via standard import, or create a new module if
    not exist.
    """

    def load_module(self, full_name):
        orig = full_name.split('.')[-1]

        if orig not in sklearn.__all__:
            raise ImportError('%s not in sklearn' % orig)

        mod = sys.modules.setdefault(full_name, imp.new_module(full_name))
        mod.__file__ = ''
        mod.__name__ = full_name
        mod.__path__ = ''
        mod.__loader__ = self
        mod.__package__ = '.'.join(full_name.split('.')[:-1])

        code = _code.substitute({'mod_name': orig})

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
