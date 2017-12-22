def to_flat_df(df):
    d = dict()
    for index, row in df.iterrows():
        for k in row.index:
            d['%s_%s' % (str(k), index)] = float(row[k])
    return d
