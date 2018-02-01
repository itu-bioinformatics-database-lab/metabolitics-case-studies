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
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn_utils.utils import SkUtilsIO
from metabolitics.preprocessing import MetaboliticsPipeline

from metabolitics_sampling import MetaboliticsSampling, SamplingDiffTransformer
from utils import sampling_to_hist


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
def save_sampling_hist():

    path = '/media/muhammedhasan/hdd/sampling-bc/'

    files = list()
    for i in os.listdir(path):
        index, label = i.split('.')[0].split('#')
        files.append((int(index), label))
    files = sorted(files, key=lambda x: x[0])

    X = list()
    y = list()

    for index, label in files:

        t = time()
        filename = '%d#%s.json.gz' % (index, label)

        with gzip.open(os.path.join(path, filename), 'rt') as f:
            df = pd.DataFrame(json.load(f))

        X.append(sampling_to_hist(df))
        y.append(label)

        print(filename, 'is done in', str(time() - t), 'sec!')

        SkUtilsIO('../outputs/bc_sampling_hist.json', gz=True).to_json(X, y)


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

        for index, label in files:

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

    X, y = SkUtilsIO('../outputs/bc_sampling_hist.json', gz=True).from_json()

    # from sklearn.feature_selection import VarianceThreshold, SelectKBest
    # from sklearn_utils.preprocessing import InverseDictVectorizer

    # vect1 = DictVectorizer(sparse=False)
    # vect2 = DictVectorizer(sparse=False)
    # vt = VarianceThreshold(0.1)
    # skb = SelectKBest(k=100)
    # pipe = Pipeline([
    #     ('sampling-diff', SamplingDiffTransformer()),
    #     ('vect-vt', vect1),
    #     ('vt', vt),
    #     ('inv_vec-vt', InverseDictVectorizer(vect1, vt)),
    #     ('vect-skb', vect2),
    #     ('skb', skb),
    #     ('inv_vec-skb', InverseDictVectorizer(vect2, skb)),
    # ])

    pipe = Pipeline([
        ('sampling-diff', SamplingDiffTransformer()),
        # ('pathway-score', MetaboliticsPipeline([
        #     'feature-selection',
        #     'pathway_transformer'
        # ])),
        ('vect', DictVectorizer(sparse=False)),
        ('pca', PCA()),
        ('clf', LogisticRegression(C=0.3e-6, random_state=43))
    ])

    # X_t = pipe.fit_transform(X, y)

    # import pdb
    # pdb.set_trace()
    from sklearn.metrics import classification_report, accuracy_score

    kf = StratifiedKFold(n_splits=10, random_state=43)

    # for train_index, test_index in kf.split(X, y):
    #     X_train = np.array(X)[train_index]
    #     X_test = np.array(X)[test_index]
    #     y_train = np.array(y)[train_index]
    #     y_test = np.array(y)[test_index]

    #     try:
    #         pipe.fit(X_train, y_train)
    #     except:
    #         import pdb
    #         pdb.set_trace()

    #     y_train_pred = pipe.predict(X_train)
    #     y_pred = pipe.predict(X_test)

    #     print('train %f' % accuracy_score(y_train, y_train_pred))
    #     print('test %f' % accuracy_score(y_test, y_pred))
    #     print(classification_report(y_test, y_pred))

    scores = cross_val_score(pipe, X, y, cv=kf,
                             n_jobs=-1, scoring='f1_micro')
    print('kfold test: %s' % scores)
    print('mean: %s' % scores.mean().round(3))
    print('std: %s' % scores.std().round(3))
