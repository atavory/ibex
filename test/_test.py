import unittest
import os
from glob import glob
import doctest

from sklearn import preprocessing
from ibex.sklearn import preprocessing as pd_preprocessing
from sklearn import pipeline
from sklearn import base
from ibex.sklearn import decomposition as pd_decomposition
from sklearn import linear_model
from ibex.sklearn import linear_model as pd_linear_model
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV
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
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.StandardScaler()
        self.assertEqual(prd, prd.fit(x, y))

    def test_transform_y(self):
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        pd_xt = pd_preprocessing.StandardScaler().fit(x, y).transform(x)
        self.assertTrue(isinstance(pd_xt, pd.DataFrame))
        xt = preprocessing.StandardScaler().fit(x, y).transform(x)
        self.assertFalse(isinstance(xt, pd.DataFrame))
        np.testing.assert_equal(pd_xt, xt)

        pd_xt = pd_preprocessing.StandardScaler().fit_transform(x, y)
        self.assertTrue(isinstance(pd_xt, pd.DataFrame))
        xt = preprocessing.StandardScaler().fit_transform(x, y)
        self.assertFalse(isinstance(xt, pd.DataFrame))
        np.testing.assert_equal(pd_xt, xt)

    def test_transform_no_y(self):
        x = pd.DataFrame({'a': [1, 2, 3]})

        pd_xt = pd_preprocessing.StandardScaler().fit(x).transform(x)
        self.assertTrue(isinstance(pd_xt, pd.DataFrame))
        xt = preprocessing.StandardScaler().fit(x).transform(x)
        self.assertFalse(isinstance(xt, pd.DataFrame))
        np.testing.assert_equal(pd_xt, xt)

        pd_xt = pd_preprocessing.StandardScaler().fit_transform(x)
        self.assertTrue(isinstance(pd_xt, pd.DataFrame))
        xt = preprocessing.StandardScaler().fit_transform(x)
        self.assertFalse(isinstance(xt, pd.DataFrame))
        np.testing.assert_equal(pd_xt, xt)

    def test_fit_predict(self):
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        pd_y_hat = pd_linear_model.LinearRegression().fit(x, y).predict(x)
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        y_hat = linear_model.LinearRegression().fit(x, y).predict(x)
        self.assertFalse(isinstance(y_hat, pd.Series))
        np.testing.assert_equal(pd_y_hat, y_hat)

    def test_fit_permute_cols(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = pd_linear_model.LinearRegression().fit(x, y)

        pd_y_hat = pred.predict(x[['b', 'a']])
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        y_hat = linear_model.LinearRegression().fit(x, y).predict(x)
        self.assertFalse(isinstance(y_hat, pd.Series))
        np.testing.assert_equal(pd_y_hat, y_hat)

    def test_fit_bad_cols(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = pd_linear_model.LinearRegression().fit(x, y)

        y_hat = pred.predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

        x.rename(columns={'a': 'c'}, inplace=True)

        with self.assertRaises(KeyError):
            pred.predict(x)

    def test_object(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = frame(linear_model.LinearRegression()).fit(x, y)

        y_hat = pred.predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

        x.rename(columns={'a': 'c'}, inplace=True)

        with self.assertRaises(KeyError):
            pred.predict(x)


class _FramePipelineTest(unittest.TestCase):
    def test_pipeline_fit(self):
        s = pd_linear_model.LinearRegression()
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        p = pipeline.make_pipeline(linear_model.LinearRegression())
        pd_p = frame(p)
        pd_p = pd_p.fit(x, y)
        y_hat = pd_p.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipeline_fit_internal_pd_stage(self):
        s = pd_linear_model.LinearRegression()
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        p = pipeline.make_pipeline(pd_linear_model.LinearRegression())
        pd_p = frame(p)
        y_hat = pd_p.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))


class _TransTest(unittest.TestCase):
    def test_trans_none(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans().fit(x).transform(x)

        # Tmp Ami
        # with self.assertRaises(KeyError):
        #   trans().transform(x)

        trans().fit_transform(x)

    def test_trans_none_cols(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans(None, 'a').fit(x)

        trans(None, ['a']).fit(x)

        trans(None, 'a').fit(x).transform(x)

        trans(None, ['a']).fit(x).transform(x)

        trans(None, 'a').fit_transform(x)

        trans(None, ['a']).fit_transform(x)

        trans(None, 'a', 'b').fit(x)

        trans(None, 'b', ['a']).fit(x)

        trans(None, 'a', 'b').fit(x).transform(x)

        trans(None, 'b', ['a']).fit(x).transform(x)

        trans(None, 'a', 'b').fit_transform(x)

        trans(None, 'b', ['a']).fit_transform(x)

    def test_trans_none_bad_cols(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        bad_x = pd.DataFrame({'a': [1, 2, 3], 'c': [30, 23, 2]})

        with self.assertRaises(KeyError):
            trans(None, 'b').fit(bad_x)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit(bad_x)

        with self.assertRaises(KeyError):
            trans(None, 'b').fit(x).transform(bad_x)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit(x).transform(bad_x)

        with self.assertRaises(KeyError):
            trans(None, 'b').fit_transform(bad_x)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit_transform(bad_x)

    def test_trans_step(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans(pd_preprocessing.StandardScaler()).fit(x).fit(x)

        trans(pd_preprocessing.StandardScaler()).fit(x).transform(x)

        trans(pd_preprocessing.StandardScaler()).fit(x).fit_transform(x)


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

        grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=10)
        grid_search.fit(self._iris[self._features], self._iris['class'])
        print(grid_search.best_estimator_)


class _FeatureUnionTest(unittest.TestCase):

    class simple_transformer(object):
        def __init__(self, col=0):
            self.col = col

        def fit_transform(self, x, y):
            xt = x.iloc[:, self.col]
            return pd.DataFrame(xt, index=x.index)

        def fit(self, x, y):
            return self

        def transform(self, x):
            xt = x.iloc[:, self.col]
            return pd.DataFrame(xt, index=x.index)

    def test_pandas_support(self):
        from ibex.sklearn import pipeline as pd_pipeline

        x = pd.DataFrame(np.random.rand(500, 10), index=range(500))
        y = x.iloc[:, 0]

        trans_list = [
            ('1', self.simple_transformer(col=0)),
            ('2', self.simple_transformer(col=1))]

        xt1 = self.simple_transformer().fit_transform(x, y)
        self.assertEqual(xt1.shape, (len(x), 1))

        feat_un = pd_pipeline.FeatureUnion(trans_list)

        xt2 = feat_un.fit_transform(x, y)
        self.assertEqual(xt2.shape, (len(x), 2))
        self.assertListEqual(list(xt2.index), list(x.index))

        feat_un.fit(x, y)
        xt3 = feat_un.transform(x)
        self.assertEqual(xt3.shape, (len(x), 2))
        self.assertListEqual(list(xt3.index), list(x.index))


class _OperatorsTest(unittest.TestCase):

    def test_pipe_fit(self):
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.StandardScaler() | \
            pd_preprocessing.StandardScaler() | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

        prd = pd_preprocessing.StandardScaler() | \
            (pd_preprocessing.StandardScaler() | \
            pd_linear_model.LinearRegression())
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipe_trans_fit(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.MinMaxScaler() | \
            trans(np.sqrt, 'a') | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipe_add_trans_fit(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.MinMaxScaler() | \
            trans() + trans(np.sqrt, 'a') | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_triple_add(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        y = pd.Series([1, 2, 3])

        prd = pd_preprocessing.MinMaxScaler() | \
            trans() + trans() + trans() | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

        prd = pd_preprocessing.MinMaxScaler() | \
            trans() + (trans() + trans()) | \
            pd_linear_model.LinearRegression()
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))


class _SKLearnTest(unittest.TestCase):
    def test_linear_model(self):
        from ibex.sklearn import linear_model

        print(linear_model.LinearRegression)
        print(linear_model.LinearRegression())


class _ExamplesTest(unittest.TestCase):
    def test_notebooks(self):
        if not _nbconvert:
            return
        return
        for f_name in os.path.join(glob(os.path.dirname(__file__))):
            with open(notebook_filename) as f:
                nb = nbformat.read(f, as_version=4)

            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

            ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})


class _VerifyDocumentTest(unittest.TestCase):
    def test(self):
        this_dir = os.path.dirname(__file__)
        doc_f_names = list(glob(os.path.join(this_dir, '../docs/source/*.rst')))
        doc_f_names += [os.path.join(this_dir, '../README.rst')]
        failed, attempted = 0, 0
        for f_name in doc_f_names:
            res = doctest.testfile(f_name, module_relative=False, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
            failed += res[0]
            attempted += res[1]
        self.assertGreater(attempted, 0)
        self.assertEqual(failed, 0)


if __name__ == '__main__':
    unittest.main()
