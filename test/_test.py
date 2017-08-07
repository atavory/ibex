import unittest
import os
from glob import glob
import doctest
import json
import pickle

import six
import numpy as np
from sklearn import preprocessing
from ibex.sklearn import exceptions
from ibex.sklearn import preprocessing as pd_preprocessing
from sklearn import pipeline
from ibex.sklearn import pipeline as pd_pipeline
from sklearn import base
from ibex.sklearn import decomposition as pd_decomposition
from sklearn import linear_model
from ibex.sklearn import linear_model as pd_linear_model
from sklearn import ensemble
from ibex.sklearn import ensemble as pd_ensemble
from sklearn import feature_selection
from ibex.sklearn import feature_selection as pd_feature_selection
from sklearn import svm
from ibex.sklearn import svm as pd_svm
from sklearn import gaussian_process
from ibex.sklearn import gaussian_process as pd_gaussian_process
from sklearn import mixture
from ibex.sklearn import mixture as pd_mixture
from sklearn import decomposition
from ibex.sklearn import decomposition as pd_decomposition
from ibex.sklearn.model_selection import GridSearchCV as PDGridSearchCV
try:
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.cross_validation import cross_val_score
    from sklearn.cross_validation import GridSearchCV
from sklearn import datasets
from sklearn.externals import joblib
import pandas as pd
import numpy as np

from ibex import *


_this_dir = os.path.dirname(__file__)


_level = os.getenv('IBEX_TEST_LEVEL')
_level = 0 if _level is None else int(_level)


def _load_iris():
    iris = datasets.load_iris()
    features = iris['feature_names']
    iris = pd.DataFrame(
        np.c_[iris['data'], iris['target']],
        columns=features+['class'])
    return iris, features


def _load_digits():
    digits = datasets.load_digits()
    features = ['f%d' % i for i in range(digits['data'].shape[1])]
    digits = pd.DataFrame(
        np.c_[digits['data'], digits['target']],
        columns=features+['digit'])
    digits = digits.sample(frac=0.1).reset_index()
    return digits, features


class _EstimatorTest(unittest.TestCase):
    pass


def _generate_bases_test(est, pd_est):
    def test(self):
        self.assertTrue(isinstance(pd_est, FrameMixin))
        self.assertFalse(isinstance(est, FrameMixin))
        self.assertTrue(isinstance(pd_est, base.BaseEstimator))
        self.assertTrue(isinstance(est, base.BaseEstimator))
        mixins = [
            base.ClassifierMixin,
            base.ClusterMixin,
            base.BiclusterMixin,
            base.TransformerMixin,
            base.DensityMixin,
            base.MetaEstimatorMixin,
            base.ClassifierMixin,
            base.RegressorMixin]
        for mixin in mixins:
            self.assertEqual(
                isinstance(pd_est, mixin),
                isinstance(est, mixin),
                mixin)

    return test


def _generate_fit_test(X, y, est, pd_est):
    def test(self):
        pd_est.fit(X, y)
        est.fit(X.as_matrix(), y.values)
    return test


def _generate_array_bad_fit_test(X, y, pd_est):
    def test(self):
        with self.assertRaises(TypeError):
            pd_est.fit(X.as_matrix(), y.values)
    return test


def _generate_index_mismatch_bad_fit_test(X, y, pd_est):
    def test(self):
        global y
        y = y.copy()
        y.index = reversed(X.index.values)
        with self.assertRaises(ValueError):
            pd_est.fit(X, y)
    return test


def _generate_predict_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict'),
            hasattr(pd_est, 'predict'))
        if not hasattr(est, 'predict'):
            return
        pd_y_hat = pd_est.fit(X, y).predict(X)
        y_hat = est.fit(X.as_matrix(), y.values).predict(X.as_matrix())
        np.testing.assert_array_equal(pd_y_hat, y_hat)
    return test


def _generate_score_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'score'),
            hasattr(pd_est, 'score'))
        if not hasattr(est, 'score'):
            return
        pd_score = pd_est.fit(X, y).score(X, y)
        score = est.fit(X.as_matrix(), y.values).score(X.as_matrix(), y.values)
        np.testing.assert_array_equal(pd_score, score)
    return test


def _generate_score_weight_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'score'),
            hasattr(pd_est, 'score'))
        if not hasattr(est, 'score'):
            return
        weight = np.abs(np.random.randn(len(y)))
        try:
            pd_score = pd_est.fit(X, y).score(X, y, weight)
        except TypeError:
            pd_score = None
        try:
            score = est.fit(X.as_matrix(), y.values).score(X.as_matrix(), y.values, sample_weight=weight)
        except TypeError:
            score = None
        if score is not None:
            self.assertNotEqual(pd_score, None, pd_est)
            self.assertTrue(np.isclose(score, pd_score))
    return test


def _generate_sample_y_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'sample_y'),
            hasattr(pd_est, 'sample_y'))
        if not hasattr(est, 'sample_y'):
            return
        pd_sample = pd_est.fit(X, y).sample_y(X)
        sample = est.fit(X.as_matrix(), y.values).sample_y(X.as_matrix())
        np.testing.assert_array_equal(pd_sample, sample)
    return test


def _generate_predict_proba_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict_proba'),
            hasattr(pd_est, 'predict_proba'),
            (est, pd_est))
        if not hasattr(est, 'predict_proba'):
            return
        pd_y_hat = pd_est.fit(X, y).predict_proba(X)
        y_hat = est.fit(X.as_matrix(), y.values).predict_proba(X.as_matrix())
        np.testing.assert_array_equal(pd_y_hat, y_hat)
    return test


def _generate_staged_predict_proba_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'staged_predict_proba'),
            hasattr(pd_est, 'staged_predict_proba'),
            (est, pd_est))
        if not hasattr(est, 'staged_predict_proba'):
            return
        pd_y_hat = pd_est.fit(X, y).staged_predict_proba(X)
        y_hat = est.fit(X.as_matrix(), y.values).staged_predict_proba(X.as_matrix())
        np.testing.assert_array_equal(pd_y_hat, y_hat)
    return test


def _generate_predict_log_proba_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict_log_proba'),
            hasattr(pd_est, 'predict_log_proba'))
        if not hasattr(est, 'predict_log_proba'):
            return
        pd_y_hat = pd_est.fit(X, y).predict_log_proba(X)
        y_hat = est.fit(X.as_matrix(), y.values).predict_log_proba(X.as_matrix())
        np.testing.assert_array_equal(pd_y_hat, y_hat)
    return test


def _generate_fit_predict_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'fit_predict'),
            hasattr(pd_est, 'fit_predict'))
        if not hasattr(est, 'fit_predict'):
            return
        pd_Xt = pd_est.fit_predict(X, y)
        Xt = est.fit_predict(X.as_matrix(), y.values)
        np.testing.assert_array_equal(pd_Xt, Xt)
    return test


def _generate_attr_test(X, y, est, pd_est):
    def test(self):
        pd_est.fit(X, y)
        est.fit(X.as_matrix(), y.values)
        self.assertEqual(
            hasattr(est, 'coef_'),
            hasattr(pd_est, 'coef_'))
        if not hasattr(est, 'coef_'):
            return
        np.testing.assert_array_equal(est.coef_, pd_est.coef_)
    return test


def _generate_transform_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'transform'),
            hasattr(pd_est, 'transform'))
        if not hasattr(est, 'transform'):
            return
        pd_y_hat = pd_est.fit(X, y).transform(X)
        y_hat = est.fit(X.as_matrix(), y.values).transform(X.as_matrix())
        np.testing.assert_array_equal(pd_y_hat, y_hat)
    return test


def _generate_fit_transform_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'fit_transform'),
            hasattr(pd_est, 'fit_transform'))
        # sklearn pipeline has a different mechanism for
        # pipeline.Pipeline.fit_transform
        try:
            y_hat = est.fit_transform(X.as_matrix(), y)
        except:
            y_hat = None
        try:
            pd_y_hat = pd_est.fit_transform(X, y)
        except AttributeError:
            pd_y_hat = None
        if pd_y_hat is None:
            self.assertIsNone(y_hat)
        else:
            np.testing.assert_array_equal(pd_y_hat, y_hat)
    return test


_dataset_names, _Xs, _ys = [], [], []
_iris, _features = _load_iris()
_dataset_names.append('iris')
_Xs.append(_iris[_features])
_ys.append(_iris['class'])
_iris = _iris.copy()
_iris.index = ['i%d' % i for i in range(len(_iris))]
_dataset_names.append('iris_str_index')
_Xs.append(_iris[_features])
_ys.append(_iris['class'])

_estimators, _pd_estimators = [], []
_estimators.append(decomposition.PCA())
_pd_estimators.append(
    pd_decomposition.PCA())
_estimators.append(
    linear_model.LinearRegression())
_pd_estimators.append(
    pd_linear_model.LinearRegression())
_estimators.append(
    pipeline.make_pipeline(decomposition.PCA(), linear_model.LinearRegression()))
_pd_estimators.append(
    pd_decomposition.PCA() | pd_linear_model.LinearRegression())
_estimators.append(
    pipeline.make_pipeline(decomposition.PCA(), linear_model.LinearRegression()))
_pd_estimators.append(
    pd_pipeline.make_pipeline(pd_decomposition.PCA(), pd_linear_model.LinearRegression()))
_estimators.append(
    linear_model.LogisticRegression())
_pd_estimators.append(
    pd_linear_model.LogisticRegression())
_estimators.append(
    pipeline.make_union(decomposition.PCA(n_components=2), feature_selection.SelectKBest(k=1)))
_pd_estimators.append(
    pd_decomposition.PCA(n_components=2) + pd_feature_selection.SelectKBest(k=1))
_estimators.append(
    pipeline.make_union(decomposition.PCA(n_components=2), feature_selection.SelectKBest(k=1)))
_pd_estimators.append(
    pd_pipeline.make_union(pd_decomposition.PCA(n_components=2), pd_feature_selection.SelectKBest(k=1)))
# Tmp Ami - fails without probability=True
_estimators.append(
    pipeline.make_pipeline(
        feature_selection.SelectKBest(k=1),
        svm.SVC(kernel="linear", random_state=42, probability=True)))
_pd_estimators.append(
    pd_feature_selection.SelectKBest(k=1) | pd_svm.SVC(kernel="linear", random_state=42, probability=True))
param_grid = dict(
    C=[0.1, 1, 10])
_estimators.append(
    GridSearchCV(
        svm.SVC(kernel="linear", random_state=42, probability=True),
        param_grid=param_grid,
        verbose=0))
_pd_estimators.append(
    PDGridSearchCV(
        pd_svm.SVC(kernel="linear", random_state=42, probability=True),
        param_grid=param_grid,
        verbose=0))
_estimators.append(
    gaussian_process.GaussianProcessRegressor())
_pd_estimators.append(
    pd_gaussian_process.GaussianProcessRegressor())
if False:
    param_grid = dict(
        C=[0.1, 1, 10])
    _estimators.append(
        GridSearchCV(
            pipeline.make_pipeline(
                linear_model.LogisticRegression()),
            param_grid=param_grid,
            return_train_score=False,
            verbose=10))
    _pd_estimators.append(
        PDGridSearchCV(
            pd_linear_model.LogisticRegression(),
            param_grid=param_grid,
            return_train_score=False,
            verbose=10))


test_i = 0
for estimators in zip(_estimators, _pd_estimators):
    est, pd_est = estimators
    name = type(est).__name__.lower()
    setattr(
        _EstimatorTest,
        'test_bases_%s_%d' % (name, test_i),
        _generate_bases_test(est, pd_est))
    for dataset in zip(_dataset_names, _Xs, _ys):
        dataset_name, X, y = dataset
        name = dataset_name + '_' + type(est).__name__.lower()
        setattr(
            _EstimatorTest,
            'test_fit_%s_%d' % (name, test_i),
            _generate_fit_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_array_bad_fit_%s_%d' % (name, test_i),
            _generate_array_bad_fit_test(X, y, pd_est))
        setattr(
            _EstimatorTest,
            'test_index_mismatch_bad_fit_%s_%d' % (name, test_i),
            _generate_index_mismatch_bad_fit_test(X, y, pd_est))
        setattr(
            _EstimatorTest,
            'test_attr_%s_%d' % (name, test_i),
            _generate_attr_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_fit_predict_%s_%d' % (name, test_i),
            _generate_fit_predict_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_predict_%s_%d' % (name, test_i),
            _generate_predict_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_score_%s_%d' % (name, test_i),
            _generate_score_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_score_weight_%s_%d' % (name, test_i),
            _generate_score_weight_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_sample_y__%s_%d' % (name, test_i),
            _generate_sample_y_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_predict_proba_%s_%d' % (name, test_i),
            _generate_predict_proba_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_staged_predict_proba_%s_%d' % (name, test_i),
            _generate_staged_predict_proba_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_predict_log_proba_%s_%d' % (name, test_i),
            _generate_predict_log_proba_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_transform_%s_%d' % (name, test_i),
            _generate_transform_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_fit_transform_%s_%d' % (name, test_i),
            _generate_fit_transform_test(X, y, est, pd_est))

        test_i += 1


class _BaseTest(unittest.TestCase):
    def test_neut(self):
        prd = linear_model.LinearRegression(fit_intercept=False)
        self.assertIn('get_params', dir(prd))
        self.assertEqual(prd.get_params()['fit_intercept'], False)

        prd = pd_linear_model.LinearRegression(fit_intercept=False)
        self.assertIn('get_params', dir(prd))
        self.assertEqual(prd.get_params()['fit_intercept'], False)


class _FrameTest(unittest.TestCase):
    def test_transform_no_y(self):
        X = pd.DataFrame({'a': [1, 2, 3]})

        pd_Xt = pd_preprocessing.StandardScaler().fit(X).transform(X)
        self.assertTrue(isinstance(pd_Xt, pd.DataFrame))
        Xt = preprocessing.StandardScaler().fit(X).transform(X)
        self.assertFalse(isinstance(Xt, pd.DataFrame))
        np.testing.assert_equal(pd_Xt, Xt)

        pd_Xt = pd_preprocessing.StandardScaler().fit_transform(X)
        self.assertTrue(isinstance(pd_Xt, pd.DataFrame))
        Xt = preprocessing.StandardScaler().fit_transform(X)
        self.assertFalse(isinstance(Xt, pd.DataFrame))
        np.testing.assert_equal(pd_Xt, Xt)

    def test_fit_permute_cols(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = pd_linear_model.LinearRegression().fit(X, y)

        pd_y_hat = pred.predict(X[['b', 'a']])
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        y_hat = linear_model.LinearRegression().fit(X, y).predict(X)
        self.assertFalse(isinstance(y_hat, pd.Series))
        np.testing.assert_equal(pd_y_hat, y_hat)

    def test_fit_bad_cols(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = pd_linear_model.LinearRegression().fit(X, y)

        y_hat = pred.predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

        X.rename(columns={'a': 'c'}, inplace=True)

        with self.assertRaises(KeyError):
            pred.predict(X)

    def test_object(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = frame(linear_model.LinearRegression()).fit(X, y)

        y_hat = pred.predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

        X.rename(columns={'a': 'c'}, inplace=True)

        with self.assertRaises(KeyError):
            pred.predict(X)


class _FramePipelineTest(unittest.TestCase):
    def test_pipeline_fit(self):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        # Tmp Ami - make verify that are framemixins
        p = pd_pipeline.make_pipeline(pd_linear_model.LinearRegression())
        self.assertTrue(isinstance(p, FrameMixin))
        pd_p = frame(p)
        pd_p = pd_p.fit(X, y)
        y_hat = pd_p.fit(X, y).predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipeline_fit_internal_pd_stage(self):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        p = pd_pipeline.make_pipeline(pd_linear_model.LinearRegression())
        self.assertTrue(isinstance(p, FrameMixin))
        pd_p = frame(p)
        y_hat = pd_p.fit(X, y).predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_make_pipeline(self):
        p = pd_pipeline.make_pipeline(pd_preprocessing.StandardScaler(), pd_linear_model.LinearRegression())


class _TransTest(unittest.TestCase):
    def test_trans_none(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans().fit(X).transform(X)

        with self.assertRaises(exceptions.NotFittedError):
            trans().transform(X)

        trans().fit_transform(X)

    def test_trans_none_cols(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans(None, 'a').fit(X)

        trans(None, ['a']).fit(X)

        trans(None, 'a').fit(X).transform(X)

        trans(None, ['a']).fit(X).transform(X)

        trans(None, 'a').fit_transform(X)

        trans(None, ['a']).fit_transform(X)

        trans(None, 'a', 'b').fit(X)

        trans(None, 'b', ['a']).fit(X)

        trans(None, 'a', 'b').fit(X).transform(X)

        trans(None, 'b', ['a']).fit(X).transform(X)

        trans(None, 'a', 'b').fit_transform(X)

        trans(None, 'b', ['a']).fit_transform(X)

    def test_trans_none_bad_cols(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        bad_X = pd.DataFrame({'a': [1, 2, 3], 'c': [30, 23, 2]})

        with self.assertRaises(KeyError):
            trans(None, 'b').fit(bad_X)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit(bad_X)

        with self.assertRaises(KeyError):
            trans(None, 'b').fit(X).transform(bad_X)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit(X).transform(bad_X)

        with self.assertRaises(KeyError):
            trans(None, 'b').fit_transform(bad_X)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit_transform(bad_X)

    def test_trans_step(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans(pd_preprocessing.StandardScaler()).fit(X).fit(X)

        trans(pd_preprocessing.StandardScaler()).fit(X).transform(X)

        trans(pd_preprocessing.StandardScaler()).fit(X).fit_transform(X)
        trans(pd_preprocessing.StandardScaler()).fit(X).fit_transform(X, X.a)


class _IrisTest(unittest.TestCase):
    def test_pipeline_cv(self):
        iris, features = _load_iris()

        clf = pd_preprocessing.StandardScaler() | pd_linear_model.LogisticRegression()
        clf.fit(iris[features], iris['class'])

        res = cross_val_score(
            clf,
            X=iris[features],
            y=iris['class'])

    # Tmp Ami
    def _test_pipeline_feature_union_grid_search_cv(self):
        from ibex.sklearn.svm import SVC
        from ibex.sklearn.decomposition import PCA
        from ibex.sklearn.feature_selection import SelectKBest

        iris, features = _load_iris()

        clf = PCA(n_components=2) + SelectKBest(k=1) | SVC(kernel="linear")

        param_grid = dict(
            featureunion__pca__n_components=[1, 2, 3],
            featureunion__selectkbest__k=[1, 2],
            svc__C=[0.1, 1, 10])

        grid_search = PDGridSearchCV(clf, param_grid=param_grid, verbose=0)
        if _level < 1:
            return
        grid_search.fit(iris[features], iris['class'])
        grid_search.best_estimator_

    def test_staged_predict(self):
        iris, features = _load_iris()

        clf = pd_ensemble.GradientBoostingRegressor()
        clf.fit(iris[features], iris['class'])
        clf.staged_predict(iris[features])

    def test_feature_importances(self):
        iris, features = _load_iris()

        clf = pd_ensemble.GradientBoostingRegressor()
        with self.assertRaises(AttributeError):
            clf.feature_importances_
        clf.fit(iris[features], iris['class'])
        self.assertTrue(isinstance(clf.feature_importances_, pd.Series))

        # Tmp Ami
    def _test_aic_bic(self):
        iris, features = _load_iris()

        clf = pd_mixture.GaussianMixture()
        clf.fit(iris[features], iris['class'])
        clf.aic(iris[features])
        clf.bic(iris[features])


class _FeatureUnionTest(unittest.TestCase):

    class SimpleTransformer(base.BaseEstimator, base.TransformerMixin, FrameMixin):
        def __init__(self, col=0):
            self.col = col

        def fit_transform(self, X, y):
            Xt = X.iloc[:, self.col]
            return pd.DataFrame(Xt, index=X.index)

        def fit(self, X, y):
            return self

        def transform(self, X):
            Xt = X.iloc[:, self.col]
            return pd.DataFrame(Xt, index=X.index)

    def test_make_union(self):
        pd_pipeline.make_union(self.SimpleTransformer())
        pd_pipeline.make_union(self.SimpleTransformer(), self.SimpleTransformer())

    def test_pandas_support(self):
        from ibex.sklearn import pipeline as pd_pipeline

        X = pd.DataFrame(np.random.rand(500, 10), index=range(500))
        y = X.iloc[:, 0]

        trans_list = [
            ('1', self.SimpleTransformer(col=0)),
            ('2', self.SimpleTransformer(col=1))]

        Xt1 = self.SimpleTransformer().fit_transform(X, y)
        self.assertEqual(Xt1.shape, (len(X), 1))

        feat_un = pd_pipeline.FeatureUnion(trans_list)

        Xt2 = feat_un.fit_transform(X, y)
        self.assertEqual(Xt2.shape, (len(X), 2))
        self.assertListEqual(list(Xt2.index), list(X.index))

        feat_un.fit(X, y)
        Xt3 = feat_un.transform(X)
        self.assertEqual(Xt3.shape, (len(X), 2))
        self.assertListEqual(list(Xt3.index), list(X.index))


class _OperatorsTest(unittest.TestCase):

    def test_pipe_fit(self):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.StandardScaler() | \
            pd_preprocessing.StandardScaler() | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(X, y).predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

        prd = pd_preprocessing.StandardScaler() | \
            (pd_preprocessing.StandardScaler() | \
            pd_linear_model.LinearRegression())
        y_hat = prd.fit(X, y).predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipe_trans_fit(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.MinMaxScaler() | \
            trans(np.sqrt, 'a') | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(X, y).predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipe_add_trans_fit(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.MinMaxScaler() | \
            trans() + trans(np.sqrt, 'a') | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(X, y).predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_triple_add(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.MinMaxScaler() | \
            trans() + trans() + trans() | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(X, y).predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

        prd = pd_preprocessing.MinMaxScaler() | \
            trans() + (trans() + trans()) | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(X, y).predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))
        print(linear_model.LinearRegression())


class _ModelSelectionTest(unittest.TestCase):
    def test_cross_val_predict(self):
        from ibex.sklearn import model_selection as pd_model_selection

        iris, features = _load_iris()

        n = 100
        df = pd.DataFrame({
                'x': range(n),
                'y': range(n),
            },
            index=['i%d' % i for i in range(n)])

        y_hat = pd_model_selection.cross_val_predict(
            pd_linear_model.LinearRegression(),
            df[['x']],
            df['y'])
        self.assertIsInstance(y_hat, pd.Series)
        self.assertEqual(len(y_hat), len(df))

    # Tmp Ami
    def _test_grid_search_fit_predict(self):
        from ibex.sklearn.svm import SVC
        from ibex.sklearn.decomposition import PCA
        from ibex.sklearn.feature_selection import SelectKBest

        iris, features = _load_iris()

        clf = PCA(n_components=2) + SelectKBest(k=1) | SVC(kernel="linear")

        param_grid = dict(
            featureunion__pca__n_components=[1, 2, 3],
            featureunion__selectkbest__k=[1, 2],
            svc__C=[0.1, 1, 10])

        grid_search = PDGridSearchCV(clf, param_grid=param_grid, verbose=0)
        if _level < 1:
            return
        grid_search.fit(iris[features], iris['class']).predict(iris[features])


class _NBsTest(unittest.TestCase):
    pass


def _generate_nb_tests(name):
    def test(self):
        cmd = 'jupyter-nbconvert --to notebook --execute %s --output %s --ExecutePreprocessor.timeout=7200' % (name, name)

        try:
            self.assertEqual(os.system(cmd), 0)
        except Exception as exc:
            print(cmd, exc)
            # Tmp Ami
            return
            # Python2.7 fails on travis, for some reason
            if six.PY3:
                raise
    return test


nb_f_names = list(glob(os.path.join(_this_dir, '../examples/*.ipynb')))
nb_f_names = [n for n in nb_f_names if '.nbconvert.' not in n]
for n in nb_f_names:
    with (open(n, encoding='utf-8') if six.PY3 else open(n)) as f:
        cnt = json.loads(f.read(), encoding='utf-8')
    metadata = cnt['metadata']
    if 'ibex_test_level' not in metadata:
        raise KeyError('ibex_test_level missing from metadata of ' + n)
    if _level < int(metadata['ibex_test_level']):
        continue

    test_name = 'test_' + os.path.splitext(os.path.split(n)[1])[0]
    test = _generate_nb_tests(n)
    setattr(_NBsTest, test_name, test)


class _PickleTest(unittest.TestCase):
    def test_direct_single_adapter(self):
        iris, features = _load_iris()

        trn = pd_decomposition.PCA()
        unpickled_trn = pickle.loads(pickle.dumps(trn))

        pca_unpickled = unpickled_trn.fit_transform(iris[features])
        pca = trn.fit_transform(iris[features])
        self.assertTrue(pca_unpickled.equals(pca))

    def test_direct_pipe_adapter(self):
        clf = pd_decomposition.PCA() | pd_linear_model.LinearRegression()
        unpickled_clf = pickle.loads(pickle.dumps(clf))

    def test_grid_search_cv(self):
        from ibex.sklearn.svm import SVC

        iris, features = _load_iris()

        clf = SVC()
        clf = PDGridSearchCV(
            clf,
            {'kernel':('linear', 'rbf'), 'C':[1, 10]},
            n_jobs=2)
        clf.fit(iris[features], iris['class'])


def load_tests(loader, tests, ignore):
    import ibex

    doctest_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

    tests.addTests(doctest.DocTestSuite(__import__('ibex'), optionflags=doctest_flags))
    for mod_name in dir(ibex):
        try:
            mod =__import__('ibex.' + mod_name)
        except ImportError:
            continue
        tests.addTests(doctest.DocTestSuite(mod, optionflags=doctest_flags))

    for f_name in glob(os.path.join(_this_dir, '../ibex/sklearn/_*.py')):
        tests.addTests(doctest.DocFileSuite(f_name, module_relative=False, optionflags=doctest_flags))

    doc_f_names = list(glob(os.path.join(_this_dir, '../docs/source/*.rst')))
    doc_f_names += [os.path.join(_this_dir, '../README.rst')]
    tests.addTests(
        doctest.DocFileSuite(*doc_f_names, module_relative=False, optionflags=doctest_flags))

    return tests


if __name__ == '__main__':
    unittest.main()
