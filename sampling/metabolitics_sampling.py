from metabolitics.analysis import MetaboliticsAnalysis
from cobra.flux_analysis.sampling import OptGPSampler


class MetaboliticsSampling(MetaboliticsAnalysis):

    def sampling_analysis(self, measurements):
        # self.add_constraint(measurements)

        import pdb
        pdb.set_trace()

        return OptGPSampler(self.model, processes=1).sample(50000)
