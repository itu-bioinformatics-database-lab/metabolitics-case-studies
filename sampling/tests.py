import unittest
from unittest import mock

import pandas as pd

from metabolitics_sampling import MetaboliticsSampling
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


if __name__ == '__main__':
    unittest.main()
