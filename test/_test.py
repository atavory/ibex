import unittest
import os
from glob import glob
import doctest
import json
import pickle

import six
import numpy as np
from sklearn import preprocessing
from ibex import frame
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
from sklearn import feature_selection
from ibex.sklearn import feature_selection as pd_feature_selection
from sklearn import neighbors
from ibex.sklearn import neighbors as pd_neighbors
from sklearn import cluster
from ibex.sklearn import cluster as pd_cluster
from sklearn import decomposition
from ibex.sklearn import decomposition as pd_decomposition
from ibex.sklearn.model_selection import GridSearchCV as PDGridSearchCV
from ibex.sklearn.model_selection import cross_val_predict as pd_cross_val_predict
try:
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_predict
except ImportError:
    from sklearn.cross_validation import cross_val_score
    from sklearn.cross_validation import GridSearchCV
    from sklearn.cross_validation import cross_val_predict
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
    frame(pd_linear_model.LinearRegression()))
_estimators.append(
    linear_model.LinearRegression())
_pd_estimators.append(
    pd_linear_model.LinearRegression())
_estimators.append(
    pipeline.make_pipeline(decomposition.PCA(), linear_model.LinearRegression()))
_pd_estimators.append(
    pd_decomposition.PCA() | pd_linear_model.LinearRegression())
_estimators.append(
    pipeline.make_pipeline(feature_selection.SelectKBest(k=2), decomposition.PCA(), linear_model.LinearRegression()))
_pd_estimators.append(
    pd_feature_selection.SelectKBest(k=2) | pd_decomposition.PCA() | pd_linear_model.LinearRegression())
_estimators.append(
    pipeline.make_pipeline(decomposition.PCA(), linear_model.LinearRegression()))
_pd_estimators.append(
    pd_pipeline.make_pipeline(pd_decomposition.PCA(), pd_linear_model.LinearRegression()))
_estimators.append(
    linear_model.LogisticRegression())
_pd_estimators.append(
    pd_linear_model.LogisticRegression())
_estimators.append(
    cluster.KMeans(random_state=42))
_pd_estimators.append(
    pd_cluster.KMeans(random_state=42))
_estimators.append(
    cluster.KMeans(random_state=42))
_pd_estimators.append(
    pickle.loads(pickle.dumps(pd_cluster.KMeans(random_state=42))))
_estimators.append(
    neighbors.KNeighborsClassifier())
_pd_estimators.append(
    pd_neighbors.KNeighborsClassifier())
_estimators.append(
    ensemble.GradientBoostingClassifier())
_pd_estimators.append(
    pd_ensemble.GradientBoostingClassifier())
_estimators.append(
    pipeline.make_union(decomposition.PCA(n_components=2), feature_selection.SelectKBest(k=1)))
_pd_estimators.append(
    pd_decomposition.PCA(n_components=2) + pd_feature_selection.SelectKBest(k=1))
_estimators.append(
    pipeline.make_union(decomposition.PCA(n_components=1), decomposition.PCA(n_components=2), feature_selection.SelectKBest(k=1)))
_pd_estimators.append(
    pd_decomposition.PCA(n_components=1) + pd_decomposition.PCA(n_components=2) + pd_feature_selection.SelectKBest(k=1))
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
param_grid = dict(
    logisticregression__C=[0.1, 1, 10])
_estimators.append(
    GridSearchCV(
        pipeline.make_pipeline(
            linear_model.LogisticRegression()),
        param_grid=param_grid,
        return_train_score=False,
        verbose=0))
_pd_estimators.append(
    PDGridSearchCV(
        pd_pipeline.make_pipeline(
            pd_linear_model.LogisticRegression()),
        param_grid=param_grid,
        return_train_score=False,
        verbose=0))


_feature_selectors, _pd_feature_selectors = [], []
_feature_selectors.append(
    feature_selection.SelectKBest(k=1))
_pd_feature_selectors.append(
    pd_feature_selection.SelectKBest(k=1))
_feature_selectors.append(
    feature_selection.SelectKBest(k=1))
_pd_feature_selectors.append(
    pickle.loads(pickle.dumps(pd_feature_selection.SelectKBest(k=1))))
_feature_selectors.append(
    feature_selection.SelectKBest(k=2))
_pd_feature_selectors.append(
    pd_feature_selection.SelectKBest(k=2))
_feature_selectors.append(
    feature_selection.SelectPercentile())
_pd_feature_selectors.append(
    pd_feature_selection.SelectPercentile())
_feature_selectors.append(
    feature_selection.SelectFdr())
_pd_feature_selectors.append(
    pd_feature_selection.SelectFdr())
_feature_selectors.append(
    feature_selection.SelectFwe())
_pd_feature_selectors.append(
    pd_feature_selection.SelectFwe())
# Tmp Ami
if False:
    _feature_selectors.append(
        feature_selection.RFE(linear_model.LogisticRegression()))
    _pd_feature_selectors.append(
        pd_feature_selection.RFE(pd_linear_model.LogisticRegression()))


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


def _generate_str_repr_test(pd_est):
    def test(self):
        # Tmp Ami - check it starts with Adapter
        self.assertEqual(str(pd_est), repr(pd_est))
    return test


def _generate_array_bad_fit_test(X, y, pd_est):
    def test(self):
        try:
            with self.assertRaises(TypeError):
                pd_est.fit(X.as_matrix(), y)
            with self.assertRaises(TypeError):
                pd_est.fit(X, y.values)
        except:
            print(pd_est)
            raise
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
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        self.assertTrue(pd_y_hat.index.equals(X.index))
        y_hat = est.fit(X.as_matrix(), y.values).predict(X.as_matrix())
        np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_score_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'score'),
            hasattr(pd_est, 'score'))
        if not hasattr(est, 'score'):
            return
        pd_score = pd_est.fit(X, y).score(X, y)
        # Tmp Ami - what is the type of score?
        score = est.fit(X.as_matrix(), y.values).score(X.as_matrix(), y.values)
        np.testing.assert_allclose(pd_score, score)
    return test


def _generate_cross_val_predict_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict'),
            hasattr(pd_est, 'predict'))
        if not hasattr(est, 'predict'):
            return
        pd_y_hat = pd_cross_val_predict(pd_est, X, y)
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        self.assertTrue(pd_y_hat.index.equals(X.index))
        y_hat = cross_val_predict(est, X.as_matrix(), y.values)
        np.testing.assert_allclose(pd_y_hat, y_hat)
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
        np.testing.assert_allclose(pd_sample, sample)
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
        self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
        self.assertTrue(pd_y_hat.index.equals(X.index))
        # Tmp Ami
        if False and hasattr(est, 'classes_'):
            # Tmp Ami - unreached
            self.assertTrue(pd_y_hat.columns.equals(pd_est.classes_), pd_est)
        y_hat = est.fit(X.as_matrix(), y.values).predict_proba(X.as_matrix())
        np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_staged_predict_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'staged_predict'),
            hasattr(pd_est, 'staged_predict'),
            (est, pd_est))
        if not hasattr(est, 'staged_predict'):
            return
        pd_y_hats = pd_est.fit(X, y).staged_predict(X)
        y_hats = est.fit(X.as_matrix(), y.values).staged_predict(X.as_matrix())
        for pd_y_hat, y_hat in zip(pd_y_hats, y_hats):
            self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
            # self.assertTrue(pd_y_hat.columns.equals(pd_est.classes_))
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_staged_predict_proba_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'staged_predict_proba'),
            hasattr(pd_est, 'staged_predict_proba'),
            (est, pd_est))
        if not hasattr(est, 'staged_predict_proba'):
            return
        pd_y_hats = pd_est.fit(X, y).staged_predict_proba(X)
        y_hats = est.fit(X.as_matrix(), y.values).staged_predict_proba(X.as_matrix())
        for pd_y_hat, y_hat in zip(pd_y_hats, y_hats):
            self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
            # self.assertTrue(pd_y_hat.columns.equals(pd_est.classes_))
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_predict_log_proba_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict_log_proba'),
            hasattr(pd_est, 'predict_log_proba'))
        if not hasattr(est, 'predict_log_proba'):
            return
        pd_y_hat = pd_est.fit(X, y).predict_log_proba(X)
        self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
        self.assertTrue(pd_y_hat.index.equals(X.index))
        # Tmp Ami
        if False and hasattr(est, 'classes_'):
            self.assertTrue(pd_y_hat.columns.equals(pd_est.classes_), pd_est)
        y_hat = est.fit(X.as_matrix(), y.values).predict_log_proba(X.as_matrix())
        np.testing.assert_allclose(pd_y_hat, y_hat)
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
        np.testing.assert_allclose(pd_Xt, Xt)
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
        np.testing.assert_allclose(est.coef_, pd_est.coef_)
    return test


def _generate_transform_test(X, y, est, pd_est):
    def test(self):
        if not hasattr(est, 'transform'):
            return
        pd_y_hat = pd_est.fit(X, y).transform(X)
        y_hat = est.fit(X.as_matrix(), y.values).transform(X.as_matrix())
        np.testing.assert_allclose(pd_y_hat, y_hat)
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
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


test_i = 0
for estimators in zip(_estimators + _feature_selectors, _pd_estimators + _pd_feature_selectors):
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
            'test_str_repr_%s_%d' % (name, test_i),
            _generate_str_repr_test(pd_est))
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
            'test_cross_val_predict_%s_%d' % (name, test_i),
            _generate_cross_val_predict_test(X, y, est, pd_est))
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
            'test_staged_predict_proba_%s_%d' % (name, test_i),
            _generate_staged_predict_test(X, y, est, pd_est))
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


def _generate_feature_selection_transform_test(X, y, pd_est):
    def test(self):
        pd_y_hat = pd_est.fit(X, y).transform(X)
        self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
        for f in pd_y_hat.columns:
            self.assertIn(f, X.columns)
    return test


class _FeatureSelectorTest(unittest.TestCase):
    pass


for estimators in zip( _feature_selectors,  _pd_feature_selectors):
    est, pd_est = estimators
    name = type(est).__name__.lower()
    for dataset in zip(_dataset_names, _Xs, _ys):
        dataset_name, X, y = dataset
        name = dataset_name + '_' + type(est).__name__.lower()
        setattr(
            _FeatureSelectorTest,
            'test_feature_selection_transform_%s_%d' % (name, test_i),
            _generate_feature_selection_transform_test(X, y, pd_est))

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


class _NBsTest(unittest.TestCase):
    pass


def _generate_nb_tests(name):
    def test(self):
        cmd = 'jupyter-nbconvert --to notebook --execute %s --output %s --ExecutePreprocessor.timeout=7200' % (name, name)

        try:
            self.assertEqual(os.system(cmd), 0)
        except Exception as exc:
            print(cmd, exc)
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


class _ExternalSubclass(pd_linear_model.LinearRegression):
    pass


class _ExternalComposer(object):
    def __init__(self):
        self.prd = pd_linear_model.LinearRegression()


class _PickleTest(unittest.TestCase):
    def test_direct_single_adapter(self):
        iris, features = _load_iris()

        trn = pd_decomposition.PCA()
        unpickled_trn = pickle.loads(pickle.dumps(trn))

        pca_unpickled = unpickled_trn.fit_transform(iris[features])
        pca = trn.fit_transform(iris[features])
        self.assertTrue(pca_unpickled.equals(pca), (pca.columns, pca_unpickled.columns))

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
            n_jobs=1)
        clf.fit(iris[features], iris['class'])

    def test_external_subclass(self):
        e = _ExternalSubclass()
        with self.assertRaises(TypeError):
            pickle.dumps(e)

    def test_external_composition(self):
        e = _ExternalComposer()
        pickle.loads(pickle.dumps(e))


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
