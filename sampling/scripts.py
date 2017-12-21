from time import time

import click
from sklearn_utils.utils import SkUtilsIO
from metabolitics.preprocessing import MetaboliticsPipeline

from metabolitics_sampling import MetaboliticsSampling


@click.group()
def cli():
    pass


@cli.command()
def save_sampling_on_bc():
    X, y = SkUtilsIO('../datasets/diseases/BC.csv').from_csv(
        label_column='stage')
    y = ['healthy' if i == 'h' else 'bc' for i in y]

    pipe = MetaboliticsPipeline(['naming', 'metabolic-standard-scaler'])
    X_t = pipe.fit_transform(X, y)

    def transform(x):
        analyzer = MetaboliticsSampling('recon2').copy()
        return analyzer.sampling_analysis(x)

    X_sampled = Parallel(n_jobs=8)(delayed(transform)(x) for x in X_t)

    SkUtilsIO('../outputs/sampling_anaylsis_bc.json', gz=True) \
        .to_json(X_sampled, y)
