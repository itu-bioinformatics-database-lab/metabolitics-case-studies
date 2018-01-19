import os
import math
import gzip
import json
import pickle
from time import time

import click
import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
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
        if i < 94:
            continue
        analyzer = MetaboliticsSampling('recon2')

        t = time()
        d = analyzer.sampling_analysis(x)
        t_end = time() - t

        path = '/media/muhammedhasan/hdd/sampling-bc/%d#%s.json.gz' % (
            i, label)
        with gzip.open(path, 'wt') as f:
            f.write(json.dumps(d.to_dict('records')))

        print('%dth analysis ended in %d sec' % (i, t_end))


@cli.command()
def sampling_train_kde():

    path = '/media/muhammedhasan/hdd/sampling-bc/'
    bandwidth = 0.1

    files = list()
    for i in os.listdir(path):
        index, label = i.split('.')[0].split('#')
        files.append((int(index), label))
    files = sorted(files, key=lambda x: x[0])

    with open('%s../sampling_anaylsis_bc.p' % path, 'wb') as fp:

        for index, label in files[107:]:

            t = time()
            filename = '%d#%s.json.gz' % (index, label)

            with gzip.open(os.path.join(path, filename), 'rt') as f:
                df = pd.DataFrame(json.load(f))

            rounding = int(-math.log10(bandwidth)) if bandwidth < 1 else 1
            kde = KernelDensity(bandwidth=bandwidth).fit(df.round(rounding))
            pickle.dump([label, kde], fp)

            print(filename, 'is done', 'in', str(time() - t), 'sec!')


@cli.command()
def sampling_f1():
    path = '/media/muhammedhasan/hdd/'

    with open(os.path.join(path, 'sampling_anaylsis_bc.p'), 'wb') as f:

        with open(os.path.join(path, 'sampling_anaylsis_bc_1.p'), 'rb') as f1:
            for l in f1:
                pickle.dump(pickle.loads(l), f)

        with open(os.path.join(path, 'sampling_anaylsis_bc_2.p'), 'rb') as f2:
            for l in f2:
                pickle.dump(pickle.loads(l), f)
