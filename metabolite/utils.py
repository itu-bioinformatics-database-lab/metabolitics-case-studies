import pandas as pd
import mwtab
from sklearn_utils.utils import SkUtilsIO


def mwtab_to_df(path):
    f = next(mwtab.read_files(path))

    id_factor_mapping = {
        i['local_sample_id']: i['factors'].split(':')[1]
        if i != 'Control' else 'healthy'
        for i in f['SUBJECT_SAMPLE_FACTORS']['SUBJECT_SAMPLE_FACTORS']
    }

    metabolite_measurements = dict()
    for i in f['MS_METABOLITE_DATA']['MS_METABOLITE_DATA_START']['DATA']:
        m = i['metabolite_name']
        del i['metabolite_name']
        metabolite_measurements[m] = i

    df = pd.DataFrame(metabolite_measurements, dtype=float)
    df.insert(0, 'labels', [id_factor_mapping[i] for i in df.index])
    df = df.reset_index()
    del df['index']

    return df
