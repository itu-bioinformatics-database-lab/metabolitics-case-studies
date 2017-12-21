from metabolitics.analysis import MetaboliticsAnalysis
from cobra.flux_analysis.sampling import OptGPSampler, ACHRSampler


class MetaboliticsSampling(MetaboliticsAnalysis):
    def sampling_analysis(self, measurements):
        self.add_constraint(measurements)
        return OptGPSampler(self.model, processes=3).sample(10000)
