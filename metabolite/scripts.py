import json
from time import time
import multiprocessing
from collections import defaultdict

import click
import numpy as np
import pandas as pd
from scipy.spatial.distance import correlation, cosine
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn_utils.utils import SkUtilsIO, map_dict_list
from sklearn_utils.preprocessing import DictInput, FeatureRenaming

from metabolitics.preprocessing import MetaboliticsPipeline, MetaboliticsTransformer
from metabolitics.utils import load_network_model, load_metabolite_mapping

from utils import mwtab_to_df, generate_complete_data


@click.group()
def cli():
    pass


@cli.command()
@click.argument('disease_name')
def analysis_and_save_disease(disease_name):
    path = '../datasets/diseases/%s.csv' % disease_name
    X, y = SkUtilsIO(path).from_csv(label_column='labels')

    pipe = MetaboliticsPipeline([
        'metabolite-name-mapping',
        'standard-scaler',
        'metabolitics-transformer',
    ])
    X_t = pipe.fit_transform(X, y)

    SkUtilsIO('../outputs/%s_analysis_with_std.json' % disease_name,
              gz=True).to_json(X_t, y)


@cli.command()
def bc_performance():
    X, y = SkUtilsIO(
        '../datasets/bc_analysis_with_std.json', gz=True).from_json()

    pipe = Pipeline([
        ('metabolitics', MetaboliticsPipeline([
            'reaction-diff',
            # 'feature-selection',
            'pathway-transformer',
        ])),
        ('vect', DictVectorizer(sparse=False)),
        ('pca', PCA()),
        ('clf', LogisticRegression(C=0.3e-6, random_state=43))
    ])

    kf = StratifiedKFold(n_splits=10, random_state=43)

    cv_score = cross_validate(pipe, X, y, cv=kf, n_jobs=-1, scoring='f1_micro')

    print(cv_score)

    import pdb
    pdb.set_trace()


@cli.command()
@click.argument('disease_name')
def analysis_mwtab(disease_name):
    df = mwtab_to_df('../datasets/diseases/%s.mwtab' % disease_name)
    df.to_csv('../outputs/%s.csv' % disease_name, index=False)


@cli.command()
def parse_naming_files():

    df = pd.read_csv(
        '../datasets/naming/recon-store-metabolites.tsv', sep='\t')

    model = load_network_model('recon2')
    mappings = defaultdict(dict)

    for i, row in df.iterrows():
        m = '%s_c' % row['abbreviation']

        if m not in model.metabolites:
            continue

        for k in row.keys()[1:]:
            if type(row[k]) == str:
                mappings[k][row[k]] = m

    for k, v in mappings.items():
        db = k.replace('Id', '')

        with open('../outputs/%s-mapping.json' % db, 'w') as f:
            json.dump(v, f)


@cli.command()
def coverage_test_metabolites():

    model = load_network_model('recon2')

    X, y = SkUtilsIO('../datasets/diseases/BC.csv') \
        .from_csv(label_column='stage')
        
    X, y = X[:24], y[:24]

    X = FeatureRenaming(load_metabolite_mapping('toy')) \
        .fit_transform(X, y)

    X = generate_complete_data(model, X, y)
    X = DictInput(StandardScaler()).fit_transform(X, y)

    X = map_dict_list(
        X, value_func=lambda k, v: v + np.random.normal(0, 0.1))

    SkUtilsIO('../outputs/coverage_test#metabolites.json',
              gz=True).to_json(X, y)


@cli.command()
def coverage_test_generate():

    model = load_network_model('recon2')

    X, y = SkUtilsIO(
        '../datasets/coverage_test/coverage_test#metabolites.json',
        gz=True).from_json()

    df = pd.DataFrame.from_records(X)
    transformer = MetaboliticsTransformer(model)

    t = time()
    X_ref = transformer.fit_transform(X, y)

    SkUtilsIO('../outputs/coverage_test#coverage=1.json',
              gz=True).to_json(X_ref, y)

    print('Ref done!')
    print(time() - t)

    for i in range(100):

        for coverage in np.linspace(0.95, 0.05, 19):

            selected_metabolite = np.random.choice(
                df.columns,
                int(np.ceil(len(model.metabolites) * coverage)),
                replace=False)

            t = time()
            X_selected = df[selected_metabolite].to_dict('records')
            X_t = transformer.fit_transform(X_selected, y)
            print(time() - t)

            name = 'coverage=%f#iteration=%d' % (coverage, i)

            SkUtilsIO('../outputs/coverage_test#%s.json' %
                      name, gz=True).to_json(X_t, y)
            print('%s done!' % name)


@cli.command()
def coverage_test_visualize():

    path = '../datasets/coverage_test'

    X, y = SkUtilsIO('%s/coverage_test_#coverage=1.json' %
                     path, gz=True).from_json()
    y = [i if i != 'h' else 'healthy'for i in y]

    pipe = Pipeline([
        ('metabolitics', MetaboliticsPipeline([
            'reaction-diff',
            'pathway-transformer',
        ])),
        ('vect', DictVectorizer(sparse=False))
    ])
    X_ref = pipe.fit_transform(X, y)

    distances = list()

    for i in range(100):

        t_distances = [1]

        for coverage in np.linspace(0.95, 0.05, 19):

            name = 'coverage=%f#iteration=%d' % (coverage, i)

            try:
                X, _ = SkUtilsIO('%s/coverage_test_#%s.json' %
                                 (path, name), gz=True).from_json()
            except FileNotFoundError:
                break

            X_t = pipe.fit_transform(X, y)

            t_distances.append(1 - np.mean([
                correlation(X_ref[i], X_t[i]) for i in range(len(X))
            ]))

            print(t_distances)

        if len(t_distances) > 1:
            distances.append(t_distances)

    avg_distances = np.mean(distances, axis=0)

    x = np.linspace(0.95, 0.05, 19)[:len(avg_distances)]
    y = avg_distances
    plt.plot(y)
    plt.xticks(range(len(y)), x)
    plt.show()
