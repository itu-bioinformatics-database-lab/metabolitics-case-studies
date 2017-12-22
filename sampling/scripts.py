from time import time

import click
from joblib import Parallel, delayed
from sklearn_utils.utils import SkUtilsIO
from metabolitics.preprocessing import MetaboliticsPipeline

from metabolitics_sampling import MetaboliticsSampling


@click.group()
def cli():
    pass

def transform(x):
    analyzer = MetaboliticsSampling('recon2')
    return analyzer.sampling_analysis(x)


@cli.command()
def save_sampling_on_bc():
    X, y = SkUtilsIO('../datasets/diseases/BC.csv').from_csv(
        label_column='stage')
    y = ['healthy' if i == 'h' else 'bc' for i in y]

    pipe = MetaboliticsPipeline(['naming', 'metabolic-standard-scaler'])
    X_t = pipe.fit_transform(X, y)


    X_sampled = list()
    for x in X_t:
        x_sample = transform(x)
        X_sampled.append(x_sample)

        SkUtilsIO('../outputs/sampling_anaylsis_bc.json', gz=True) \
            .to_json(X_sampled, y)
