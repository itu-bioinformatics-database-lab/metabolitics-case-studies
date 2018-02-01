from typing import List, Dict
from itertools import chain

import numpy as np
from cobra.flux_analysis.sampling import OptGPSampler
from sklearn_utils.utils import filter_by_label
from metabolitics.analysis import MetaboliticsAnalysis
from metabolitics.preprocessing import ReactionDiffTransformer

from utils import sum_probabilities, is_dict_finite


class MetaboliticsSampling(MetaboliticsAnalysis):

    def sampling_analysis(self, measurements):
        self.add_constraint(measurements)
        return OptGPSampler(self.model, processes=8).sample(10000)


ReactionCdf = Dict[str, Dict[int, int]]


class SamplingDiffTransformer(ReactionDiffTransformer):
    '''Converts sampled kde functions to diff scores.'''

    def fit(self, X: List[ReactionCdf], y):
        '''
        :param X: list of reaction cdfs.
        :param y: list of labels
        '''
        self.ref_ = dict()

        healthies = filter_by_label(X, y, self.reference_label)[0]

        for r in self.model.reactions:
            # TODO: investigate why there is nan values in there
            reaction_hists = [
                x[r.id]
                for x in healthies
                if r.id in x and is_dict_finite(x[r.id])
            ]

            if reaction_hists:
                self.ref_[r.id] = sum_probabilities(reaction_hists)

        return self

    def transform(self, X, y=None):
        '''
        :param X: list of sklearn KDE objects
        '''
        return [{
            reaction.id: self._reaction_cdf_diff(reaction.id, x)
            for reaction in self.model.reactions
            if reaction.id in x and reaction.id in self.ref_ and is_dict_finite(x[reaction.id])
        } for x in X]

    def _reaction_cdf_diff(self, reaction_id: str, x: Dict):
        r_ref = self.ref_[reaction_id]
        r_x = x[reaction_id]
        cdf1, cdf2, diff = 0, 0, 0

        for k in sorted(set(chain(r_x.keys(), r_ref.keys()))):
            cdf1 += r_x.get(k, 0)
            cdf2 += r_ref.get(k, 0)
            diff += cdf2 - cdf1

        assert np.isfinite(diff)
        return diff
