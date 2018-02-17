import click
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

from sklearn_utils.utils import SkUtilsIO
from sklearn_utils.preprocessing import InverseDictVectorizer
from metabolitics.preprocessing import MetaboliticsPipeline


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
