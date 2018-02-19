import click
import mwtab
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn_utils.utils import SkUtilsIO
from sklearn_utils.preprocessing import InverseDictVectorizer
from metabolitics.preprocessing import MetaboliticsPipeline

from utils import mwtab_to_df


@click.group()
def cli():
    pass


@cli.command()
def save_bc_with_std():
    X, y = SkUtilsIO('../datasets/diseases/BC.csv').from_csv(
        label_column='stage')
    y = ['healthy' if i == 'h' else 'bc' for i in y]

    vect = DictVectorizer(sparse=False)
    pipe = Pipeline([
        ('naming', MetaboliticsPipeline(['naming'])),
        ('vect', vect),
        ('std', StandardScaler()),
        ('inv_vec-standard', InverseDictVectorizer(vect)),
        ('fva', MetaboliticsPipeline(['metabolitics-transformer'])),
    ])
    X_t = pipe.fit_transform(X, y)

    SkUtilsIO('../outputs/bc_analysis_with_std.json',
              gz=True).to_json(X_t, y)


@cli.command()
def bc_performance():
    X, y = SkUtilsIO(
        '../datasets/bc_analysis_with_std.json', gz=True).from_json()

    pipe = Pipeline([
        ('metabolitics', MetaboliticsPipeline([
            'reaction-diff',
            'feature-selection',
            'pathway_transformer'
        ])),
        ('vect', DictVectorizer(sparse=False)),
        ('pca', PCA()),
        ('clf', LogisticRegression(C=0.3e-6, random_state=43))
    ])

    cv_score = cross_validate(pipe, X, y, cv=10, n_jobs=-1, scoring='f1_micro')

    import pdb
    pdb.set_trace()


@cli.command()
def analysis_mwtab():

    mwtab_to_df('../datasets/diseases/crohn.mwtab')

    import pdb
    pdb.set_trace()
