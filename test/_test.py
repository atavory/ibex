import unittest
import os
from glob import glob
import doctest
import subprocess

import six
from sklearn import preprocessing
from ibex.sklearn import preprocessing as pd_preprocessing
from sklearn import pipeline
from ibex.sklearn import pipeline as pd_pipeline
from sklearn import base
from ibex.sklearn import decomposition as pd_decomposition
from sklearn import linear_model
from ibex.sklearn import linear_model as pd_linear_model
from sklearn import ensemble
from ibex.sklearn import ensemble as pd_ensemble
from sklearn import mixture
from ibex.sklearn import mixture as pd_mixture
from ibex.sklearn.model_selection import GridSearchCV as PDGridSearchCV
try:
    from sklearn.model_selection import cross_val_score
except ImportError:
    from sklearn.cross_validation import cross_val_score
from sklearn import datasets
import pandas as pd
import numpy as np
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    _nbconvert = True
except ImportError:
    _nbconvert = False

from ibex import *


_this_dir = os.path.dirname(__file__)


class _ConceptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prd = pd_linear_model.LinearRegression()
        cls._clf = pd_linear_model.LogisticRegression()

    def test_base_estimator(self):
        self.assertTrue(
            isinstance(self._prd, base.BaseEstimator))
        self.assertTrue(
            isinstance(self._clf, base.BaseEstimator))

    def test_regressor_mixin(self):
        self.assertTrue(
            isinstance(self._prd, base.RegressorMixin))
        self.assertTrue(
            isinstance(self._clf, base.ClassifierMixin))

    def test_classifier_mixin(self):
        self.assertFalse(
            isinstance(self._prd, base.ClassifierMixin))
        self.assertFalse(
            isinstance(self._clf, base.RegressorMixin))


class _BaseTest(unittest.TestCase):
    def test_neut(self):
        prd = linear_model.LinearRegression(fit_intercept=False)
        self.assertIn('get_params', dir(prd))
        self.assertEqual(prd.get_params()['fit_intercept'], False)

        prd = pd_linear_model.LinearRegression(fit_intercept=False)
        self.assertIn('get_params', dir(prd))
        self.assertEqual(prd.get_params()['fit_intercept'], False)

    def test_getattr(self):
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        prd = pd_linear_model.LinearRegression()
        with self.assertRaises(AttributeError):
            prd.coef_
        prd.fit(x, y)
        prd.coef_


class _FrameTest(unittest.TestCase):
    def test_fit(self):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.StandardScaler()
        self.assertEqual(prd, prd.fit(X, y))

    def test_transform_y(self):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        pd_Xt = pd_preprocessing.StandardScaler().fit(X, y).transform(X)
        self.assertTrue(isinstance(pd_Xt, pd.DataFrame))
        Xt = preprocessing.StandardScaler().fit(X, y).transform(X)
        self.assertFalse(isinstance(Xt, pd.DataFrame))
        np.testing.assert_equal(pd_Xt, Xt)

        pd_Xt = pd_preprocessing.StandardScaler().fit_transform(X, y)
        self.assertTrue(isinstance(pd_Xt, pd.DataFrame))
        Xt = preprocessing.StandardScaler().fit_transform(X, y)
        self.assertFalse(isinstance(Xt, pd.DataFrame))
        np.testing.assert_equal(pd_Xt, Xt)

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

    def test_fit_predict(self):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        pd_y_hat = pd_linear_model.LinearRegression().fit(X, y).predict(X)
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        y_hat = linear_model.LinearRegression().fit(X, y).predict(X)
        self.assertFalse(isinstance(y_hat, pd.Series))
        np.testing.assert_equal(pd_y_hat, y_hat)

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

        # Tmp Ami
        # with self.assertRaises(KeyError):
        #   trans().transform(X)

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


class _IrisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        iris = datasets.load_iris()
        cls._features = iris['feature_names']
        cls._iris = pd.DataFrame(
            np.c_[iris['data'], iris['target']],
            columns=cls._features+['class'])

    def test_fit_transform(self):
        decomp = trans(pd_decomposition.PCA(n_components=2), None, ['pc1', 'pc2'])

        tr = decomp.fit_transform(self._iris)
        self.assertEqual(set(tr.columns), set(['pc1', 'pc2']))

    def test_fit_plus_transform(self):
        decomp = trans(pd_decomposition.PCA(n_components=2), None, ['pc1', 'pc2'])

        tr = decomp.fit(self._iris).transform(self._iris)
        self.assertEqual(set(tr.columns), set(['pc1', 'pc2']))

    def test_logistic_regression_cv(self):
        clf = pd_linear_model.LogisticRegression()
        clf.fit(self._iris[self._features], self._iris['class'])

        res = cross_val_score(
            clf,
            X=self._iris[self._features],
            y=self._iris['class'])

    def test_predict_proba(self):
        clf = pd_linear_model.LogisticRegression()
        clf.fit(self._iris[self._features], self._iris['class'])

        clf.predict_proba(self._iris[self._features])

    def test_predict_log_proba(self):
        clf = pd_linear_model.LogisticRegression()
        clf.fit(self._iris[self._features], self._iris['class'])

        clf.predict_log_proba(self._iris[self._features])

    def test_pipeline_cv(self):
        clf = pd_preprocessing.StandardScaler() | pd_linear_model.LogisticRegression()
        clf.fit(self._iris[self._features], self._iris['class'])

        res = cross_val_score(
            clf,
            X=self._iris[self._features],
            y=self._iris['class'])

    def test_pipeline_feature_union_grid_search_cv(self):
        from ibex.sklearn.svm import SVC
        from ibex.sklearn.decomposition import PCA
        from ibex.sklearn.feature_selection import SelectKBest

        clf = PCA(n_components=2) + SelectKBest(k=1) | SVC(kernel="linear")

        param_grid = dict(
            featureunion__pca__n_components=[1, 2, 3],
            featureunion__selectkbest__k=[1, 2],
            svc__C=[0.1, 1, 10])

        grid_search = PDGridSearchCV(clf, param_grid=param_grid, verbose=10)
        grid_search.fit(self._iris[self._features], self._iris['class'])
        grid_search.best_estimator_

    def test_staged_predict(self):
        clf = pd_ensemble.GradientBoostingRegressor()
        clf.fit(self._iris[self._features], self._iris['class'])
        clf.staged_predict(self._iris[self._features])

    def test_feature_importances(self):
        clf = pd_ensemble.GradientBoostingRegressor()
        with self.assertRaises(AttributeError):
            clf.feature_importances_
        clf.fit(self._iris[self._features], self._iris['class'])
        self.assertTrue(isinstance(clf.feature_importances_, pd.Series))

        # Tmp Ami
    def _test_aic_bic(self):
        clf = pd_mixture.GaussianMixture()
        clf.fit(self._iris[self._features], self._iris['class'])
        clf.aic(self._iris[self._features])
        clf.bic(self._iris[self._features])


class _DigitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        digits = datasets.load_digits()
        cls._features = ['f%d' % i for i in range(digits['data'].shape[1])]
        cls._digits = pd.DataFrame(
            np.c_[digits['data'], digits['target']],
            columns=cls._features+['digit'])
        cls._digits = cls._digits.sample(frac=0.1).reset_index()

    def test_cv(self):
        clf = pd_decomposition.PCA() | pd_linear_model.LogisticRegression()

        n_components = [20, 40, 64]
        Cs = np.logspace(-4, 4, 3)

        estimator = PDGridSearchCV(
            clf,
            {'pca__n_components': [20, 40, 64], 'logisticregression__C': np.logspace(-4, 4, 3)})

        estimator.fit(self._digits[self._features], self._digits.digit)


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


class _SKLearnTest(unittest.TestCase):
    def test_linear_model(self):
        from ibex.sklearn import linear_model

        print(linear_model.LinearRegression)
        print(linear_model.LinearRegression())


class _ModelSelectionTest(unittest.TestCase):
    def test_cross_val_predict(self):
        from ibex.sklearn import model_selection as pd_model_selection

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
        self.assertEquals(len(y_hat), len(df))


class _ExamplesTest(unittest.TestCase):
        def test_nbs(self):
            nb_f_names = list(glob(os.path.join(_this_dir, '../examples/*.ipynb')))
            nb_f_names = [n for n in nb_f_names if '.nbconvert.' not in n]
            for n in nb_f_names:
                cmd = 'jupyter nbconvert --to notebook --execute %s' % n
                try:
                    subprocess.check_call(cmd.split(' '))
                except Exception as exc:
                    print(exc.output)
                    # Tmp Ami - add flags for this
                    # Python2.7 fails on travis, for some reason
                    if six.PY3:
                        raise


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
    # Tmp Ami
    mod =__import__('ibex.sklearn.model_selection')
    tests.addTests(doctest.DocTestSuite(mod, optionflags=doctest_flags))
    mod =__import__('ibex.sklearn.pipeline')
    test = tests.addTests(doctest.DocTestSuite(mod, optionflags=doctest_flags))
    from ibex.sklearn import pipeline
    tests.addTests(doctest.DocTestSuite(pipeline, optionflags=doctest_flags))

    doc_f_names = list(glob(os.path.join(_this_dir, '../docs/source/*.rst')))
    doc_f_names += [os.path.join(_this_dir, '../README.rst')]
    test = tests.addTests(
        doctest.DocFileSuite(*doc_f_names, module_relative=False, optionflags=doctest_flags))

    return tests


if __name__ == '__main__':
    unittest.main()
