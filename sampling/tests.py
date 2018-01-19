import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from metabolitics_sampling import MetaboliticsSampling, SamplingDiffTransformer
from scripts import *


class TestMetaboliticsTransformer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MetaboliticsSampling('textbook')

    def test_sampling_analysis(self):
        df = self.analyzer.sampling_analysis({'h_c': 1})


class TestUtils(unittest.TestCase):
    def test_to_flat_df(self):
        df = pd.DataFrame([{'d': 1}, {'d': 2}])
        flat_df = to_flat_df(df)
        self.assertEqual(flat_df, {'d_0': 1.0, 'd_1': 2.0})


class TestSamplingDiffTransformer(unittest.TestCase):

    def setUp(self):
        self.N = 10000
        self.transformer = SamplingDiffTransformer('healthy', self.N)
        self.kdes = [
            KernelDensity().fit(np.ones((self.N, 2))),
            KernelDensity().fit(np.ones((self.N, 2)) * 2)
        ]
        self.y = ['healthy', 'x']

    def test_fit(self):
        self.transformer.fit(self.kdes, self.y)
        self.assertAlmostEquals(
            self.transformer.ref_kde.sample(self.N).mean(), 1, places=1)

    def test_transform(self):
        diff_scores = self.transformer.fit_transform(self.kdes, self.y)

        for i in range(2):
            self.assertLess(diff_scores[0][i], 0.1)
            self.assertLess(diff_scores[0][i], diff_scores[1][i])


if __name__ == '__main__':
    unittest.main()
