from metabolitics.analysis import MetaboliticsAnalysis
from cobra.flux_analysis.sampling import OptGPSampler


class MetaboliticsSampling(MetaboliticsAnalysis):

    def sampling_analysis(self, measurements):
        self.add_constraint(measurements)
        return OptGPSampler(self.model, processes=24).sample(10000)
