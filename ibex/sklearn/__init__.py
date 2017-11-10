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
import traceback
import operator

import six
import sklearn
import pandas as pd

from .._base import Pipeline as PdPipeline
from .._base import FeatureUnion as PdFeatureUnion


__all__ = sklearn.__all__


_sklearn_ver = int(sklearn.__version__.split('.')[1])


def _pipeline_make_pipeline(*estimators):
    """
    Creates a pipeline from estimators.

    Arguments:

        transformers: Iterable of estimators.

    Returns:

        A :class:`ibex.sklearn.pipeline.Pipeline` object.

    Example:

        >>> from ibex.sklearn import preprocessing
        >>> from ibex.sklearn import linear_model
        >>> from ibex.sklearn import pipeline
        >>>
        >>> pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
        Pipeline(...)

    """
    estimators = list(estimators)

    if len(estimators) > 1:
        return six.moves.reduce(operator.or_, estimators[1:], estimators[0])

    name = type(estimators[0]).__name__.lower()
    return PdPipeline([(name, estimators[0])])


def _pipeline_make_union(*transformers):
    """
    Creates a union from transformers.

    Arguments:

        transformers: Iterable of transformers.

    Returns:

        A :class:`ibex.sklearn.pipeline.FeatureUnion` object.

    Example:

        >>> from ibex.sklearn import preprocessing as pd_preprocessing
        >>> from ibex.sklearn import pipeline as pd_pipeline

        >>> trn = pd_pipeline.make_union(
        ...     pd_preprocessing.StandardScaler(),
        ...     pd_preprocessing.MaxAbsScaler())

    """

    transformers = list(transformers)

    if len(transformers) > 1:
        return six.moves.reduce(operator.add, transformers[1:], transformers[0])

    name = type(transformers[0]).__name__.lower()
    return PdFeatureUnion([(name, transformers[0])])


def _cluster_transform(self, base_ret):
    """

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.cluster import KMeans as PdKMeans

        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> iris[features]
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        ...

        >>> clt = PdKMeans(n_clusters=3, random_state=1).fit(iris[features])
        >>> clt.transform(iris[features])
            0         1         2
        0    3.419251  0.146942  5.059542
        1    3.398574  0.438169  5.114943
        2    3.569357  0.412301  5.279355
        3    3.422410  0.518837  5.153590
        4    3.467264  0.197970  5.104334
        ...

    """

    if isinstance(base_ret, pd.DataFrame):
        base_ret.columns = list(range(len(base_ret.columns)))
    return base_ret


def _cluster_fit_transform(self, base_ret):
    """

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.cluster import KMeans as PdKMeans

        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> iris[features]
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        ...

        >>> clt = PdKMeans(n_clusters=3, random_state=1).fit(iris[features])
        >>> clt.transform(iris[features])
            0         1         2
        0    3.419251  0.146942  5.059542
        1    3.398574  0.438169  5.114943
        2    3.569357  0.412301  5.279355
        3    3.422410  0.518837  5.153590
        4    3.467264  0.197970  5.104334
        ...

    """
    if isinstance(base_ret, pd.DataFrame):
        base_ret.columns = list(range(len(base_ret.columns)))
    return base_ret


def _replace(orig, name):
    if orig == 'pipeline':
        if name ==  'Pipeline':
            return PdPipeline
        if name == 'FeatureUnion':
            return PdFeatureUnion


def _add(orig):
    if orig == 'pipeline':
        return {
            'make_union': _pipeline_make_union,
            'make_pipeline': _pipeline_make_pipeline,
        }

    return {}


_X = pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]})
_y = pd.Series([1, 0, 1])


def _regression_coef_(self, base_ret):
    """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LinearRegression as PdLinearRegression

        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> iris[features]
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        ...

        >>> prd =  PdLinearRegression().fit(iris[features], iris['class'])
        >>>
        >>> prd.coef_
        sepal length (cm)   -0.109741
        sepal width (cm)    -0.044240
        petal length (cm)    0.227001
        petal width (cm)     0.609894
        dtype: float64
        >>>
        >>> prd.intercept_
        0.19208...

    Example:

        >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression

        >>> clf =  PdLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...           0.414988          1.461297          -2.262141         -1.029095
        1...           0.416640         -1.600833           0.577658         -1.385538
        2...          -1.707525         -1.534268           2.470972          2.555382
        >>> clf.intercept_
        0    0.265606
        1    1.085424
        2   -1.214715
        dtype: float64

    """
    if len(base_ret.shape) == 1:
        return pd.Series(base_ret, index=self.x_columns)

    if len(base_ret.shape) == 2:
        index = self.y_columns
        return pd.DataFrame(base_ret, index=index, columns=self.x_columns)

    raise RuntimeError()


def _regression_intercept_(self, base_ret):
    """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LinearRegression as PdLinearRegression

        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> iris[features]
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        ...

        >>> prd =  PdLinearRegression().fit(iris[features], iris['class'])
        >>>
        >>> prd.coef_
        sepal length (cm)   -0.109741
        sepal width (cm)    -0.044240
        petal length (cm)    0.227001
        petal width (cm)     0.609894
        dtype: float64
        >>>
        >>> prd.intercept_
        0.19208...

    Example:

        >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression

        >>> clf =  PdLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...           0.414988          1.461297          -2.262141         -1.029095
        1...           0.416640         -1.600833           0.577658         -1.385538
        2...          -1.707525         -1.534268           2.470972          2.555382
        >>> clf.intercept_
        0    0.265606
        1    1.085424
        2   -1.214715
        dtype: float64

    """

    # Tmp Ami - replace next by is_nummeric or is_scalar
    if isinstance(base_ret, (type(1), type(1.), type(1 + 1j))):
        return base_ret

    if len(base_ret.shape) == 1:
        return pd.Series(base_ret)

    raise RuntimeError()


def _classification_coef_(self, base_ret):
    """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LinearRegression as PdLinearRegression

        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> iris[features]
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        ...

        >>> prd =  PdLinearRegression().fit(iris[features], iris['class'])
        >>>
        >>> prd.coef_
        sepal length (cm)   -0.109741
        sepal width (cm)    -0.044240
        petal length (cm)    0.227001
        petal width (cm)     0.609894
        dtype: float64
        >>>
        >>> prd.intercept_
        0.19208...

    Example:

        >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression

        >>> clf =  PdLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...           0.414988          1.461297          -2.262141         -1.029095
        1...           0.416640         -1.600833           0.577658         -1.385538
        2...          -1.707525         -1.534268           2.470972          2.555382
        >>> clf.intercept_
        0    0.265606
        1    1.085424
        2   -1.214715
        dtype: float64

    """
    if len(base_ret.shape) == 1:
        return pd.Series(base_ret, index=self.x_columns)

    if len(base_ret.shape) == 2:
        index = self.classes_
        return pd.DataFrame(base_ret, index=index, columns=self.x_columns)

    raise RuntimeError()


def _classification_intercept_(self, base_ret):
    """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LinearRegression as PdLinearRegression

        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> iris[features]
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        ...

        >>> prd =  PdLinearRegression().fit(iris[features], iris['class'])
        >>>
        >>> prd.coef_
        sepal length (cm)   -0.109741
        sepal width (cm)    -0.044240
        petal length (cm)    0.227001
        petal width (cm)     0.609894
        dtype: float64
        >>>
        >>> prd.intercept_
        0.19208...

    Example:

        >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression

        >>> clf =  PdLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...           0.414988          1.461297          -2.262141         -1.029095
        1...           0.416640         -1.600833           0.577658         -1.385538
        2...          -1.707525         -1.534268           2.470972          2.555382
        >>> clf.intercept_
        0    0.265606
        1    1.085424
        2   -1.214715
        dtype: float64

    """

    # Tmp Ami - replace next by is_nummeric or is_scalar
    if isinstance(base_ret, (type(1), type(1.), type(1 + 1j))):
        return base_ret

    if len(base_ret.shape) == 1:
        return pd.Series(base_ret)

    raise RuntimeError()


def _feature_importances_(self, base_ret):
    """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.ensemble import RandomForestClassifier as PdRandomForestClassifier

        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> iris[features]
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        ...

        >>> clf =  PdRandomForestClassifier(random_state=42).fit(iris[features], iris['class'])
        >>>
        >>> clf.feature_importances_
        sepal length (cm)    0.129268
        sepal width (cm)     0.015822
        petal length (cm)    0.444740
        petal width (cm)     0.410169
        dtype: float64

    """
    return pd.Series(base_ret, index=self.x_columns)


def _get_estimator_extras(orig, est):
    orig_attrs = set(dir(est()))
    try:
        final_attrs = set(dir(est().fit(_X, _y)))
    except TypeError:
        try:
            final_attrs = set(dir(est().fit(_X)))
        except ValueError:
            final_attrs = set(dir(est().fit(_y)))
    final_attrs = final_attrs.union(orig_attrs)
    final_attrs = [a for a in final_attrs if not a.startswith('_')]
    final_attrs = [a for a in final_attrs if not a.startswith('n_')]
    attrs = {}
    methods = {}

    is_classifier = issubclass(est, sklearn.base.ClassifierMixin)
    is_clusterer = issubclass(est, sklearn.base.ClusterMixin)

    if is_clusterer:
        methods['transform'] = _cluster_transform
        methods['fit_transform'] = _cluster_fit_transform

    if 'intercept_' in final_attrs:
        if is_classifier:
            attrs['intercept_'] = _classification_intercept_
        else:
            attrs['intercept_'] = _regression_intercept_
    if 'coef_' in final_attrs:
        if is_classifier:
            attrs['coef_'] = _classification_coef_
        else:
            attrs['coef_'] = _regression_coef_
    if 'feature_importances_' in final_attrs:
        attrs['feature_importances_'] = _feature_importances_
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
        extras = ibex.sklearn._get_estimator_extras('$mod_name', est)
        extra_attribs = extras['attrs']
        extra_methods = extras['methods']
    except:
        extra_attribs = {}
        extra_methods = {}
        # _traceback.print_exc()
        _sys.stderr.write(str(est))

    try:
        globals()[name] = ibex.frame_ex(
            getattr(_orig, est.__name__),
            extra_attribs=extra_attribs,
            extra_methods=extra_methods)
    except TypeError as e:
        # _traceback.print_exc()
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
        if orig == 'decomposition':
            from ._decomposition import update_module as _decomposition_update_module
            _decomposition_update_module(mod)
        if orig == 'feature_selection':
            from ._feature_selection import update_module as _feature_selection_update_module
            _feature_selection_update_module(mod)
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

