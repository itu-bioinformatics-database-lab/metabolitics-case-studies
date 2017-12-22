import json
from time import time

import click
from joblib import Parallel, delayed
from sklearn_utils.utils import SkUtilsIO
from metabolitics.preprocessing import MetaboliticsPipeline

from metabolitics_sampling import MetaboliticsSampling
from utils import to_flat_df


@click.group()
def cli():
    pass


@cli.command()
def save_sampling_on_bc():
    X, y = SkUtilsIO('../datasets/diseases/BC.csv').from_csv(
        label_column='stage')
    y = ['healthy' if i == 'h' else 'bc' for i in y]

    def transform(x):
        analyzer = MetaboliticsSampling('recon2')
        return analyzer.sampling_analysis(x)

    pipe = MetaboliticsPipeline(['naming', 'metabolic-standard-scaler'])
    X_t = pipe.fit_transform(X, y)

    with open('../outputs/sampling_anaylsis_bc.json', 'w') as f:
        for x, label in zip(X_t, y):
            d = to_flat_df(transform(x))
            f.write('%s\n' % json.dumps([label, d]))
