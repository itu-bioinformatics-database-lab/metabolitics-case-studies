from time import time
import gzip
import json


import click
# from joblib import Parallel, delayed
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

    for i, (x, label) in enumerate(zip(X_t, y)):
        analyzer = MetaboliticsSampling('recon2')

        t = time()
        d = analyzer.sampling_analysis(x)
        t_end = time() - t

        path = '../outputs/sampling-bc/%d#%s.json.gz' % (i, label)
        with gzip.open(path, 'wt') as f:
            f.write(json.dumps(d.to_dict('records')))

        print('%dth analysis ended in %d sec' % (i, t_end))


@cli.command()
def sampling_diff():
    X, y = SkUtilsIO('../datasets/diseases/BC.csv').from_csv(
        label_column='stage')

    import pdb
    pdb.set_trace()

    # f = open('/media/muhammedhasan/hdd/sampling_anaylsis_bc.json')
