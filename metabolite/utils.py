import pandas as pd
import mwtab
from sklearn_utils.utils import SkUtilsIO


def mwtab_to_df(path, id_mapping='PubChem ID'):
    '''
    Parse mwtab file to df

    :param path: path of mwtab file
    :param id_mapping: which db will be used to annotate metabolite names.
    Those are valid inputs {None, 'PubChem ID', 'KEGG ID', 'HMDB'}
    '''
    f = next(mwtab.read_files(path))

    id_factor_mapping = {
        i['local_sample_id']: i['factors'].split('|')[1].split(':')[1].strip()
        if i['factors'].split('|')[1].strip() != 'Factor3:Healthy' else 'healthy'
        for i in f['SUBJECT_SAMPLE_FACTORS']['SUBJECT_SAMPLE_FACTORS']
    }

    metabolites_names = {
        i['metabolite_name']: i[id_mapping]
        for i in f['METABOLITES']['METABOLITES_START']['DATA']
        if id_mapping in i
    }

    metabolite_measurements = dict()
    for i in f['MS_METABOLITE_DATA']['MS_METABOLITE_DATA_START']['DATA'][1:]:
        m = i['']
        del i['']
        if id_mapping:
            if m in metabolites_names:
                metabolite_measurements[metabolites_names[m]] = i
        else:
            metabolite_measurements[m] = i

    df = pd.DataFrame(metabolite_measurements, dtype=float)
    df.insert(0, 'labels', [id_factor_mapping[i] for i in df.index])
    df = df.reset_index()
    del df['index']

    return df[df['labels'] != 'NA']
