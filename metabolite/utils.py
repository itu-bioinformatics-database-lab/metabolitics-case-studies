import pandas as pd
import numpy as np

import mwtab
import metabolitics.extensions


def mwtab_to_df(path, id_mapping='pubchem_id'):
    '''
    Parse mwtab file to df

    :param path: path of mwtab file
    :param id_mapping: which db will be used to annotate metabolite names.
    Those are valid inputs {None, 'PubChem ID', 'KEGG ID', 'HMDB'}
    '''
    f = next(mwtab.read_files(path))

    id_factor_mapping = {
        i['local_sample_id']: i['factors'].split('-')[1].strip()
        if not i['factors'].startswith('Source:Method Blanks') else 'healthy'
        for i in f['SUBJECT_SAMPLE_FACTORS']['SUBJECT_SAMPLE_FACTORS']
    }

    metabolites_names = {
        i['metabolite_name']: i[id_mapping]
        for i in f['METABOLITES']['METABOLITES_START']['DATA']
        if id_mapping in i and i[id_mapping]
    }

    metabolite_measurements = dict()

    for i in f['MS_METABOLITE_DATA']['MS_METABOLITE_DATA_START']['DATA']:
        m = i['metabolite_name']
        del i['metabolite_name']

        if id_mapping:
            if m in metabolites_names:
                metabolite_measurements[metabolites_names[m]] = i
        else:
            metabolite_measurements[m] = i

    df = pd.DataFrame(metabolite_measurements, dtype=float)
    df = df[[i in id_factor_mapping for i in df.index]]

    labels = [id_factor_mapping[i] for i in df.index]
    labels = [i if i != 'Adenocarcnoma' else 'Adenocarcinoma' for i in labels]

    df.insert(0, 'labels', labels)
    df = df.reset_index()
    del df['index']

    return df.replace('', np.nan).dropna(axis=1, how='any')


def generate_complete_data(model, X, y):

    def r_id_set(m_id):
        return set([
            r.id for r in model.metabolites.get_by_id(m_id).producers()
        ])

    for i, x in enumerate(X):

        measured_metabolites = set(x.keys())

        unmeasured_metabolites = set([m.id for m in model.metabolites])
        unmeasured_metabolites -= measured_metabolites

        for m in unmeasured_metabolites:
            x[m] = x[max(measured_metabolites,
                         key=lambda k: len(r_id_set(k) & r_id_set(m)))]

        print(i)

    return X
