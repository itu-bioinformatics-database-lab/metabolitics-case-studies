import unittest

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from metabolitics_sampling import MetaboliticsSampling, SamplingDiffTransformer
from utils import *
from scripts import *


class TestMetaboliticsTransformer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MetaboliticsSampling('textbook')

    def test_sampling_analysis(self):
        df = self.analyzer.sampling_analysis({'h_c': 1})
        self.assertIsNotNone(df)


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.d = [{5: 0.5, 10: 0.5}, {0: 0.5, 5: 0.5}]
        self.expected_d = {0: 0.5, 5: 1, 10: 0.5}

    def test_add_dict(self):
        self.assertEqual(add_dict(*self.d), self.expected_d)

    def test_sum_dict(self):
        self.assertEqual(sum_dict(self.d), self.expected_d)

    def test_normalize_dict(self):
        nd = normalize_dict(add_dict(*self.d))
        expected_nd = {0: 0.25, 5: 0.5, 10: 0.25}
        self.assertEqual(nd, expected_nd)

    def test_to_flat_df(self):
        df = pd.DataFrame([{'d': 1}, {'d': 2}])
        flat_df = to_flat_df(df)
        self.assertEqual(flat_df, {'d_0': 1.0, 'd_1': 2.0})

    def test_sampling_matrix_to_hist(self):
        df = pd.DataFrame([{'a': 1, 'b': 12}, {'a': 6, 'b': 17}])
        hist = {'a': {0: 0.5, 5: 0.5}, 'b': {10: 0.5, 15: 0.5}}
        self.assertEqual(sampling_to_hist(df), hist)


class TestSamplingDiffTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = SamplingDiffTransformer()

        self.X = [
            {'TAXOLte': {0: 0.5, 5: 0.5}, 'GLUDym': {10: 0.5, 15: 0.5}},
            {'TAXOLte': {5: 0.5, 10: 0.5}, 'GLUDym': {5: 0.5, 10: 0.5}},
            {'TAXOLte': {0: 0.5, 5: 0.5}, 'GLUDym': {0: 0.5, 5: 0.5}}
        ]
        self.y = ['x', 'healthy', 'healthy']

    def test_fit(self):
        self.transformer.fit(self.X, self.y)
        expected_ref = {
            'TAXOLte': {0: 0.25, 5: 0.5, 10: 0.25},
            'GLUDym': {0: 0.25, 5: 0.5, 10: 0.25}
        }
        self.assertEqual(self.transformer.ref_, expected_ref)

    def test_transform(self):
        diff_scores = self.transformer.fit_transform(self.X, self.y)

        self.assertEqual(diff_scores[0]['TAXOLte'], -0.5)
        self.assertEqual(diff_scores[0]['GLUDym'], 1.5)


if __name__ == '__main__':
    unittest.main()
