from typing import List

import numpy as np
from scipy.stats import ks_2samp
from sklearn.base import TransformerMixin
from sklearn.neighbors.kde import KernelDensity
from sklearn_utils.utils import filter_by_label
from cobra.flux_analysis.sampling import OptGPSampler
from metabolitics.analysis import MetaboliticsAnalysis


class MetaboliticsSampling(MetaboliticsAnalysis):

    def sampling_analysis(self, measurements):
        self.add_constraint(measurements)
        return OptGPSampler(self.model, processes=8).sample(10000)


class SamplingDiffTransformer(TransformerMixin):
    '''Converts sampled kde functions to diff scores.'''

    def __init__(self, reference_label, n_sample, bandwidth=1):
        """
        :param reference_label: the label diff will be performed by.
        :param n_sample: number of sample will be generated from KDEs.
        :param bandwidth: bandwidth of kde (for more check sklearn doc).
        """
        self.reference_label = reference_label
        self.n_sample = n_sample
        self.bandwidth = bandwidth

    def fit(self, X: List[KernelDensity], y):
        '''
        :param X: list of sklearn KDE objects
        :param y: list of labels
        '''
        ref_kdes = filter_by_label(X, y, self.reference_label)[0]
        self.ref_kde = KernelDensity(bandwidth=self.bandwidth).fit(
            np.vstack(kde.sample(self.n_sample) for kde in ref_kdes))
        return self

    def transform(self, X: List[KernelDensity]):
        '''
        :param X: list of sklearn KDE objects
        '''
        samples = map(lambda kde: kde.sample(self.n_sample), X)
        ref_sample = self.ref_kde.sample(self.n_sample)

        return [[ks_2samp(*t).statistic for t in zip(ref_sample.T, x.T)]
                for x in samples]
