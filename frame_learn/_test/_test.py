import unittest

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import pipeline, base
import pandas as pd
import numpy as np


from frame_learn import *


class _BaseTest(unittest.TestCase):
    def test_is_subclass(self):
        self.assertFalse(
            _Step.is_subclass(
                linear_model.LinearRegression()))
        self.assertTrue(
            _Step.is_subclass(
                frame(linear_model.LinearRegression())))

    def test_bases(self):
        prd = frame(linear_model.LinearRegression())
        clf = frame(linear_model.LogisticRegression())

        self.assertTrue(
            isinstance(prd, base.BaseEstimator))
        self.assertTrue(
            isinstance(clf, base.BaseEstimator))

        self.assertTrue(
            isinstance(prd, base.RegressorMixin))
        self.assertTrue(
            isinstance(clf, base.ClassifierMixin))

        self.assertFalse(
            isinstance(prd, base.ClassifierMixin))
        self.assertFalse(
            isinstance(clf, base.RegressorMixin))

    def test_attr(self):
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        prd = frame(linear_model.LinearRegression())
        prd.fit(x, y).coef_

class _PDframeTest(unittest.TestCase):
    def test_transform_y(self):
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        xt = frame(preprocessing.StandardScaler()).fit(x, y).transform(x)
        self.assertTrue(isinstance(xt, pd.DataFrame))
        xt = frame(preprocessing.StandardScaler()).fit_transform(x, y)
        self.assertTrue(isinstance(xt, pd.DataFrame))

    def test_transform_no_y(self):
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        xt = frame(preprocessing.StandardScaler()).fit(x).transform(x)
        self.assertTrue(isinstance(xt, pd.DataFrame))
        xt = frame(preprocessing.StandardScaler()).fit_transform(x)
        self.assertTrue(isinstance(xt, pd.DataFrame))

    def test_fit(self):
        s = frame(linear_model.LinearRegression())
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        y_hat = frame(linear_model.LinearRegression()).fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_fit_permute_cols(self):
        s = frame(linear_model.LinearRegression())
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = frame(linear_model.LinearRegression()).fit(x, y)

        y_hat = pred.predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

        y_hat = pred.predict(x[list(reversed(list(x.columns)))])
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipeline_fit(self):
        s = frame(linear_model.LinearRegression())
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        p = pipeline.make_pipeline(linear_model.LinearRegression())
        y_hat = frame(p).fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipeline_fit_internal_pd_stage(self):
        s = frame(linear_model.LinearRegression())
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        p = pipeline.make_pipeline(frame(linear_model.LinearRegression()))
        y_hat = frame(p).fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))


class _ApplyTest(unittest.TestCase):
    def test_apply_none(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        apply().transform(x)

    def test_apply_none_cols(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        apply(columns=['a']).transform(x)

    def test_apply_none_bad_cols(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        with self.assertRaises(KeyError):
            apply(columns=['c']).transform(x)

    def test_apply_pd(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        apply(lambda x: pd.DataFrame({'sqrt_a': np.sqrt(x['a'])})).transform(x)

    def test_apply_no_pd_single(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        apply({'sqrt_a': lambda x: np.sqrt(x)}, columns='a').transform(x)

    def test_apply_no_pd_multiple(self):
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        print(apply({('sqrt_a', 'sqrt_b'): lambda x: np.sqrt(x)}, columns=['a', 'b']).transform(x))


class _UnionTest(unittest.TestCase):

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
        """ simple test to make sure the pandas wrapper works properly"""
        x = pd.DataFrame(np.random.rand(500, 10), index=range(500))
        y = x.iloc[:, 0]

        trans_list = [
            ('1', self.simple_transformer(col=0)),
            ('2', self.simple_transformer(col=1))]

        xt1 = self.simple_transformer().fit_transform(x, y)
        self.assertEqual(xt1.shape, (len(x), 1))

        feat_un = FeatureUnion(trans_list)

        xt2 = feat_un.fit_transform(x, y)
        self.assertEqual(xt2.shape, (len(x), 2))
        self.assertListEqual(list(xt2.index), list(x.index))

        feat_un.fit(x, y)
        xt3 = feat_un.transform(x)
        self.assertEqual(xt3.shape, (len(x), 2))
        self.assertListEqual(list(xt3.index), list(x.index))


class _OperatorsTest(unittest.TestCase):

    def test_pipe_fit(self):
        s = frame(linear_model.LinearRegression())
        x = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        prd = frame(preprocessing.StandardScaler()) | \
            frame(preprocessing.StandardScaler()) | \
            frame(linear_model.LinearRegression())
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipe_apply_fit(self):
        s = frame(linear_model.LinearRegression())
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        y = pd.Series([1, 2, 3])

        prd = frame(preprocessing.MinMaxScaler()) | \
            apply({'sqrt_a': np.sqrt, 'sqr_a': lambda x: x ** 2}, columns='a') | \
            frame(linear_model.LinearRegression())
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))

    def test_pipe_add_apply_fit(self):
        s = frame(linear_model.LinearRegression())
        x = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        y = pd.Series([1, 2, 3])

        prd = frame(preprocessing.MinMaxScaler()) | \
            apply() + apply({'sqrt_a': np.sqrt, 'sqr_a': lambda x: x ** 2}, columns='a') | \
            frame(linear_model.LinearRegression())
        y_hat = prd.fit(x, y).predict(x)
        self.assertTrue(isinstance(y_hat, pd.Series))


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
        """ simple test to make sure the pandas wrapper works properly"""
        x = pd.DataFrame(np.random.rand(500, 10), index=range(500))
        y = x.iloc[:, 0]

        trans_list = [
            ('1', self.simple_transformer(col=0)),
            ('2', self.simple_transformer(col=1))]

        xt1 = self.simple_transformer().fit_transform(x, y)
        self.assertEqual(xt1.shape, (len(x), 1))

        feat_un = FeatureUnion(trans_list)

        xt2 = feat_un.fit_transform(x, y)
        self.assertEqual(xt2.shape, (len(x), 2))
        self.assertListEqual(list(xt2.index), list(x.index))

        feat_un.fit(x, y)
        xt3 = feat_un.transform(x)
        self.assertEqual(xt3.shape, (len(x), 2))
        self.assertListEqual(list(xt3.index), list(x.index))


if __name__ == '__main__':
    unittest.main()
