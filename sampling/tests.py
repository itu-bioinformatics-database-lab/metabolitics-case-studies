import unittest
from metabolitics_sampling import MetaboliticsSampling


class TestMetaboliticsTransformer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MetaboliticsSampling('textbook')

    def test_sampling_analysis(self):
        df = self.analyzer.sampling_analysis({'h_c': 1})
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    unittest.main()
