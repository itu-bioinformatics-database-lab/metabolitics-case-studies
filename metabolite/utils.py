import mwtab


def mwtab_to_df(path):
    f = next(mwtab.read_files(path))

    id_factor_mapping = {
        i['local_sample_id']: i['factors']
        for i in f['SUBJECT_SAMPLE_FACTORS']['SUBJECT_SAMPLE_FACTORS']
    }

    metabolite_measurements = dict()
    for i in f['MS_METABOLITE_DATA']['MS_METABOLITE_DATA_START']['DATA']:
        m = i['metabolite_name']
        del i['metabolite_name']
        metabolite_measurements[m] = i

    import pdb
    pdb.set_trace()

    f['MS_METABOLITE_DATA']['MS_METABOLITE_DATA_START']['samples']
