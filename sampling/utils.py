from typing import List, Dict
from collections import Counter
from functools import reduce

import numpy as np
import pandas as pd
from sklearn_utils.utils import map_dict


def to_flat_df(df):
    d = dict()
    for index, row in df.iterrows():
        for k in row.index:
            d['%s_%s' % (str(k), index)] = float(row[k])
    return d


def sampling_to_hist(df: pd.DataFrame, bin_size=5):
    '''
    :param sampling: pandas dataframe of sampling result.
    '''
    def hist_f(v):
        density, bins = np.histogram(
            v, bins=int(2000 // bin_size), density=True, range=(-1000, 1000))
        unity_density = density / density.sum()
        return unity_density, bins

    return {
        reaction: {k: v for v, k in zip(*hist_f(df[reaction])) if v}
        for reaction in df
    }


def is_dict_finite(d: Dict) -> bool:
    return np.isfinite(list(d.values())).all()


def add_dict(d1: Dict, d2: Dict) -> Dict:
    c = Counter(d1)
    c.update(Counter(d2))
    return dict(c)


def sum_dict(ds: List[Dict]) -> Dict:
    return reduce(add_dict, ds)


def normalize_dict(d: Dict) -> Dict:
    s = sum(d.values())
    return map_dict(d, value_func=lambda k, v: v / s)


def sum_probabilities(ds: List[Dict]) -> Dict:
    return normalize_dict(sum_dict(ds))
