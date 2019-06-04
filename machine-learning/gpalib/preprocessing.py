#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Library with functions for data preprocessing
"""

from datetime import datetime
import json

import numpy as np
import pandas as pd

import pickle

from sklearn.preprocessing import StandardScaler

from gpalib.analysis import contracts_with_many_okpd


# ======================================================================================
# Data aggregation after shortening OKPD variable
# Data flattening (one contract = one observation)
# ======================================================================================

def shorten_okpd_(data: pd.DataFrame, okpd_sym=2, debug=True):
    """Creation of column with shortened OKPD"""

    df = data.copy()

    okpd_column_name = 'okpd{}'.format(okpd_sym)
    df[okpd_column_name] = df.okpd.apply(lambda a: str(a)[:okpd_sym])
    unique_okpd = np.unique(df[okpd_column_name])

    if debug:
        print('New column `{}` is added'.format(okpd_column_name))

    return df, okpd_column_name, unique_okpd


def agg_stats_for_okpd_(data: pd.DataFrame, okpd_column_name: str, unique_okpds: list, debug=True):
    """
    Aggregation for columns `okpd_cntr_num` and `okpd_good_cntr_num` for shortened OKPD
    """

    df = data.copy()

    df_ = df.drop_duplicates('okpd')
    agg_df = (
        df_[['okpd_cntr_num', 'okpd_good_cntr_num', okpd_column_name]]
            .groupby(okpd_column_name)[['okpd_cntr_num', 'okpd_good_cntr_num']]
            .agg('sum')
    )

    for row in agg_df.itertuples():
        df.loc[df[okpd_column_name] == row.Index, 'okpd_cntr_num'] = row.okpd_cntr_num
        df.loc[df[okpd_column_name] == row.Index, 'okpd_good_cntr_num'] = row.okpd_good_cntr_num

    if debug:
        print('Data for columns `okpd_cntr_num` and `okpd_good_cntr_num` is aggregated')

    return df


def agg_stats_for_sup_(data: pd.DataFrame, okpd_column_name: str, unique_okpds: list, debug=True):
    """
    Aggregation for column `sup_okpd_cntr_num` for shortened OKPD
    """

    df = data.copy()

    df_ = df.drop_duplicates(['supID', 'okpd'])
    agg_df = (
        df_[['sup_okpd_cntr_num', okpd_column_name, 'supID']]
            .groupby(['supID', okpd_column_name])['sup_okpd_cntr_num']
            .agg('sum').to_dict()
    )

    res = []
    for row in df.itertuples():
        res.append([agg_df[(row.supID, getattr(row, okpd_column_name))]])

    df.loc[:, 'sup_okpd_cntr_num'] = res

    if debug:
        print('Data for column `sup_okpd_cntr_num` is aggregated\n')

    return df


def aggregate_data_by_shortened_okpd(data: pd.DataFrame, okpd_sym=2, debug=True):
    """
    Aggragating data after shortening OKPD variable. 
    Shortening means deletting symbols in `okpd` variable after first N chars. 
    """

    data1, okpd_column_name, unique_okpd = shorten_okpd_(data, okpd_sym, debug=debug)

    data2 = agg_stats_for_okpd_(data1, okpd_column_name, unique_okpd, debug=debug)

    data3 = agg_stats_for_sup_(data2, okpd_column_name, unique_okpd, debug=debug)

    return data3


def create_new_var_based_on_okpd_(data: pd.DataFrame, okpd_column_name, debug=True):
    df = data.copy()

    # Column name swap due to mistake
    tmp_name = 'tmp_name'
    df = df.rename(columns={'okpd_good_cntr_num': tmp_name})
    df = df.rename(columns={'okpd_cntr_num': 'okpd_good_cntr_num'})
    df = df.rename(columns={tmp_name: 'okpd_cntr_num'})

    # New variable calculation
    df['okpd_good_cntr_share'] = df.okpd_good_cntr_num / df.okpd_cntr_num
    df['sup_okpd_cntr_share'] = df.sup_okpd_cntr_num / df.sup_cntr_num

    # New variable: number of initial 9-symbols OKPD per contract
    cntrIDs, cntr_many_okpd = contracts_with_many_okpd(df, debug=False)
    cntrID_cv = cntr_many_okpd.cntrID.value_counts()
    df['okpd_num'] = df.cntrID.apply(lambda cntr_id: cntrID_cv.get(cntr_id, 1))

    if debug:
        print('New variable `okpd_num` was created')

    return df


def get_dummies_for_okpd_vars_(data: pd.DataFrame, okpd_column_name: str, debug=True):
    """
    Getting dummy variables for variables depending on OKPD: 
    aggregated okpd variable and `sup_okpd_cntr_share`
    """
    df = data.copy()

    okpd_dummies = pd.get_dummies(
        df[okpd_column_name],
        prefix='{}'.format(okpd_column_name)
    )

    sup_okpd_cntr_share_dummies = pd.get_dummies(
        df[okpd_column_name],
        prefix='socs'
    )

    if debug:
        print('Dummy variables for `{}` and `sup_okpd_contract_share` were created'.format(okpd_column_name))

    return pd.concat([df, okpd_dummies, sup_okpd_cntr_share_dummies], axis=1)


def flatten_agg_okpds_(data: pd.DataFrame, okpd_column_name, debug=True):
    """Aggregation of data about many OKPD of contract in one observation"""

    df = data.copy()
    res = []

    # cntrIDs with several OKPD
    cntrIDs, _ = contracts_with_many_okpd(df, debug=False)

    # If there are no contracts with several OKPD
    if not cntrIDs:
        return df

    # dummy columns with OKPD
    okpd_columns = [col for col in df.columns if okpd_column_name + '_' in col]

    # OKPD aggregation in `res` variable
    for idx, cntrID in enumerate(cntrIDs):
        okpd_agg = np.sum(
            df.loc[lambda d: d.cntrID == cntrID, okpd_columns[0]:okpd_columns[-1]],
            axis=0).values

        res.append(okpd_agg.tolist())

    # Only one row for unique cntrID
    df = df.drop_duplicates('cntrID')

    # Splitting df on contracts with one and many OKPD
    df_one_okpd = df[~df.cntrID.isin(cntrIDs)]
    df_many_okpd = df[df.cntrID.isin(cntrIDs)]

    # Custom sorting of rows in order as cntrID in `cntrIDs`)
    df_many_okpd = (
        df_many_okpd
            .set_index('cntrID')
            .reindex(cntrIDs)
            .reset_index()
    )

    # Updating values of dummy columns with OKPD
    df_many_okpd.loc[lambda df: df.cntrID.isin(cntrIDs), okpd_columns[0]:okpd_columns[-1]] = np.array(res)

    # Concatenation of two dataframes
    df = pd.concat([df_one_okpd, df_many_okpd], sort=False)

    if debug:
        print('Data was flattened: one contract = one row')

    return df.sample(frac=1).reset_index(drop=True)


def transform_okpd_good_share_(data: pd.DataFrame, data_with_dup: pd.DataFrame, okpd_column_name: str, debug=True):
    """
    Transformation of `okpd_good_cntr_share` for cases, when contract included several OKPD.
    Adding min, mean and max share over presented OKPD in contract.
    """

    df = data.copy()

    cntrIDs, _ = contracts_with_many_okpd(data_with_dup, debug=False)

    # Share of good contracts for every aggregated OKPD
    okpd_to_share = (
        data_with_dup.drop_duplicates(okpd_column_name)[[okpd_column_name, 'okpd_good_cntr_share']]
            .set_index(okpd_column_name)
            .to_dict()
        ['okpd_good_cntr_share']
    )

    # Good contracts shares of OKPD presented in contract
    cntr_okpd = {}
    for row in data_with_dup.itertuples():
        if cntr_okpd.get(row.cntrID, None):
            cntr_okpd[row.cntrID].append(okpd_to_share[getattr(row, okpd_column_name)])
        else:
            cntr_okpd[row.cntrID] = [okpd_to_share[getattr(row, okpd_column_name)]]

    # Adding new variables
    df['okpd_good_share_min'] = df.cntrID.apply(lambda cntr_id: np.min(cntr_okpd[cntr_id]))
    df['okpd_good_share_mean'] = df.cntrID.apply(lambda cntr_id: np.mean(cntr_okpd[cntr_id]))
    df['okpd_good_share_max'] = df.cntrID.apply(lambda cntr_id: np.max(cntr_okpd[cntr_id]))

    # Deleting initial variable
    df = df.drop(['okpd_good_cntr_share'], axis=1)

    if debug:
        print('New variables (min, mean, max) instead of `okpd_good_cntr_share` were created')

    return df


def transform_sup_okpd_cntr_share_(data: pd.DataFrame, okpd_column_name: str, debug=True):
    """Shares of contracts for supplier over all OKPD in one row"""

    df = data.copy()

    sup_agg_okpd = (
        df.drop_duplicates(['supID', okpd_column_name])
            .groupby('supID')[okpd_column_name]
            .apply(list)
            .to_dict()
    )

    sup_agg_okpd_share = (
        df.drop_duplicates(['supID', okpd_column_name])
            .groupby('supID')['sup_okpd_cntr_share']
            .apply(list)
            .to_dict()
    )

    # Restructuring data in following format: 
    # {supID1: {okpd1: sup_okpd_share1, okpd2: sup_okpd_share2, ...}, 
    # supID2: ...}
    res_sup_agg = {key: dict(zip(sup_agg_okpd[key], sup_agg_okpd_share[key])) for key in sup_agg_okpd}

    socs = [clmn for clmn in df.columns if 'socs' in clmn]
    clms = list(df.columns)

    # Variable for storing values for `socs_` columns
    res = []

    # Variable for storing `socs_` values for supplier, whose values were already calculated
    memory = {}

    # Indexes of `socs_` columns
    socs_column_idx = {
        okpd: clms.index('socs_{}'.format(okpd)) - clms.index(socs[0])
        for okpd in np.unique(df[okpd_column_name])
    }

    # Calculating values for `socs_` columns for every row of dataset
    df = df.sort_values(by='supID')
    for row in df['supID'].iteritems():
        if row[1] in memory:
            res.append(memory[row[1]])
            continue

        okpds_and_shares = res_sup_agg[row[1]]
        ar = np.zeros(len(socs)).tolist()
        for okpd in okpds_and_shares:
            ar[socs_column_idx[okpd]] = okpds_and_shares[okpd]

        memory[row[1]] = ar
        res.append(ar)

    # Updating values of `socs_` columns in dataset
    df.iloc[:, clms.index(socs[0]):clms.index(socs[-1]) + 1] = np.array(res)

    if debug:
        print('`socs_` variables were updated\n')

    return df.sample(frac=1).reset_index(drop=True)


def flatten_data(data: pd.DataFrame, okpd_column_name: str, debug=True):
    """Flattening data so that one row was describing one contract"""

    # Adding new variables before flattening
    data1 = create_new_var_based_on_okpd_(data, okpd_column_name, debug=debug)

    # Delete duplications after OKPD aggregation
    data1 = data1.drop_duplicates(['cntrID', okpd_column_name])

    # Get dummy variables for variables depending on OKPD
    data1 = get_dummies_for_okpd_vars_(data1, okpd_column_name, debug=debug)

    # Flattening data: one row = one contract
    data2 = flatten_agg_okpds_(data1, okpd_column_name, debug=debug)

    # Transforming `okpd_good_cntr_share` for flattened data
    data3 = transform_okpd_good_share_(data2, data1, okpd_column_name, debug=debug)

    # Transforming `sup_okpd_cntr_share` for flattened data
    data4 = transform_sup_okpd_cntr_share_(data3, okpd_column_name, debug=debug)

    return data4


# ======================================================================================
# Feature engineering
# ======================================================================================

def feature_eng_new_time_vars_(data: pd.DataFrame, debug=True):
    df = data.copy()

    # Planned length of contracts in days
    cntr_start = pd.to_datetime(df['sign_date'], format="%Y%m%d")
    cntr_end = pd.to_datetime(df['exec_date'], format="%Y%m%d")
    df['plan_cntr_len'] = (cntr_end - cntr_start).apply(lambda time_delta: time_delta.days)

    # Price per day
    df['day_price'] = df.price / df.plan_cntr_len

    # Month of contract signing
    df['sign_month'] = df['sign_date'].astype(str).str[4:6].astype(int)

    # Quarter of contract signing
    df['sign_quarter'] = df.sign_month.apply(lambda a: (a - 1) // 3 + 1)

    if debug:
        print('4 new variables were created: `plan_cntr_len`, '
              '`day_price`, `sign_month`, `sign_quarter`. '
              'Data shape: {}'.format(df.shape))

    return df


def feature_eng_new_share_vars_(data: pd.DataFrame, debug=True):
    df = data.copy()

    df['sup_good_cntr_share'] = df.sup_good_cntr_num / df.sup_cntr_num
    df['sup_fed_cntr_share'] = df.sup_fed_cntr_num / df.sup_cntr_num
    df['sup_sub_cntr_share'] = df.sup_sub_cntr_num / df.sup_cntr_num
    df['sup_mun_cntr_share'] = df.sup_mun_cntr_num / df.sup_cntr_num

    df['org_good_cntr_share'] = df.org_good_cntr_num / df.org_cntr_num
    df['org_fed_cntr_share'] = df.org_fed_cntr_num / df.org_cntr_num
    df['org_sub_cntr_share'] = df.org_sub_cntr_num / df.org_cntr_num
    df['org_mun_cntr_share'] = df.org_mun_cntr_num / df.org_cntr_num

    if debug:
        print(
            '8 new variables were created: `sup_good_cntr_share`, `sup_fed_cntr_share`, '
            '`sup_sub_cntr_share` and `sup_mun_cntr_share` and the same for customer. '
            'Data shape: {}'.format(df.shape))

    return df


def drop_useless_columns_(data: pd.DataFrame, okpd_column_name: str, debug=True):
    df = data.copy()

    columns_to_drop = [
        # Delete useless IDs
        'valID', 'supID', 'orgID', 'okpdID', 'cntr_reg_num',

        # Delete `okpd` and shortened variant of OKPD
        'okpd', okpd_column_name,

        # Delete variables transformed in share variables
        'sup_good_cntr_num', 'sup_fed_cntr_num', 'sup_sub_cntr_num',
        'sup_mun_cntr_num', 'sup_okpd_cntr_num',

        # Delete `sup_okpd_cntr_share` due to `socs_` variables existence
        'sup_okpd_cntr_share',

        # Delete variables transformed in share variables
        'org_good_cntr_num', 'org_fed_cntr_num', 'org_sub_cntr_num', 'org_mun_cntr_num',

        # Delete `okpd_good_cntr_num` due to `okpd_good_share_` min, mean, max
        'okpd_good_cntr_num',

        # Delete `sign_date` and `exec_date` => `plan_cntr_len`, `sign_month`, `sign_quarter`
        'sign_date', 'exec_date']

    df.drop(columns_to_drop, axis=1, inplace=True)

    if debug:
        print('{} variables were deleted: {}. Data shape: {}'.format(
            len(columns_to_drop),
            ', '.join(['`{}`'.format(clmn) for clmn in columns_to_drop]),
            df.shape))

    return df


def reorder_columns_(data: pd.DataFrame, okpd_column_name: str, debug=True):
    df = data.copy()

    df = df.rename(columns={'okpd_num': 'cntr_okpd_num'})

    # Reorder columns: `cntr_result` is last column
    columns_new_order = (
        ['cntrID'] +
        [clm for clm in df.columns if clm.startswith('sup')] +
        [clm for clm in df.columns if clm.startswith('org')] +
        ['cntr_num_together'] +
        [clm for clm in df.columns if clm.startswith('okpd_')] +
        [
            'sign_month', 'sign_quarter', 'cntr_okpd_num', 'plan_cntr_len',
            'day_price', 'purch_type', 'cntr_lvl', 'price_higher_pmp',
            'price_too_low', 'pmp', 'price'
        ] +
        [clm for clm in df.columns if clm.startswith(okpd_column_name)] +
        [clm for clm in df.columns if clm.startswith('socs')] +
        ['cntr_result']
    )

    if debug:
        print('Columns were reordered\n')

    return df[columns_new_order]


def basic_feature_engineering(data: pd.DataFrame, okpd_column_name: str, debug=True):
    df = feature_eng_new_time_vars_(data, debug=debug)
    df = feature_eng_new_share_vars_(df, debug=debug)
    df = drop_useless_columns_(df, okpd_column_name, debug=debug)

    # Transform object-type columns to float-type
    for clmn in ('sup_cntr_avg_penalty_share', 'sup_sim_price_share', 'org_sim_price_share'):
        df[clmn] = df[clmn].str.replace(',', '.').astype(float)

    return reorder_columns_(df, okpd_column_name, debug=debug)


# ======================================================================================
# Preprocessing for model development based on findings from "1. exploratory-data-analysis.ipynb"
# ======================================================================================

def update_null_values_(data: pd.DataFrame):
    """Update NaN and undefined values"""

    df = data.copy()

    for var in ['purch_type', 'cntr_lvl']:
        df.loc[df[var] == -1, var] = df[var].mode()[0]

    df.loc[df['org_ter'].isna(), 'org_ter'] = df['org_ter'].mode()[0]
    df.org_ter = df.org_ter.astype(int)

    return df


def drop_useless_variables_(data: pd.DataFrame, num_var01: list, num_var: list, cat_bin_var: list, cat_var: list):
    """Deleting useless variable"""

    useless_vars = [
        'sup_1s_sev',
        'sup_1s_org_sev',
        'org_1s_sev',
        'org_1s_sup_sev',
        'org_fed_cntr_share',
        'org_sub_cntr_share',
        'org_mun_cntr_share',
        'price_higher_pmp',
        'price_too_low',
        'sup_ter',
        'pmp'
    ]

    # Updating list of available variables
    for var in useless_vars:
        if var in num_var01:
            num_var01.remove(var)
        elif var in num_var:
            num_var.remove(var)
        elif var in cat_bin_var:
            cat_bin_var.remove(var)
        else:
            cat_var.remove(var)

    return data.drop(useless_vars + ['cntrID'], axis=1)


def drop_correlating_variables_(data: pd.DataFrame, num_var01: list, num_var: list, cat_bin_var: list, cat_var: list):
    """Drop correlating varibles to exclude multicollinearity"""

    # TODO: remove less correlating variables via extra nonliner transformations
    correlating_vars = [
        'okpd_good_share_mean',
        'okpd_good_share_max',
        'price',
        'org_running_cntr_num',
        'sup_cntr_num',
        'sign_quarter'
    ]

    # Updating list of available variables
    for var in correlating_vars:
        if var in num_var01:
            num_var01.remove(var)
        elif var in num_var:
            num_var.remove(var)
        elif var in cat_bin_var:
            cat_bin_var.remove(var)
        else:
            cat_var.remove(var)

    return data.drop(correlating_vars, axis=1)


def fix_day_price_error(data: pd.DataFrame):
    """Excluding inf values for `day_price`"""

    # Delete contracts where price is equal to 0
    data = data.drop(data[data.price == 0].index)

    # If `plan_cntr_len` == 0, then `plan_cntr_len` = 1
    data.plan_cntr_len = data.plan_cntr_len.clip(lower=1)

    # Updating `day_price` variable
    data.day_price = data.price / data.plan_cntr_len

    return data


def apply_logarithmic_transformation_(data: pd.DataFrame, num_var: list):
    """Apply logarithmic transformation for quantitative variables"""

    df = data.copy()

    for var in num_var:
        # Set the lowest value equal to 1
        df[var] = df[var].clip(lower=1)

        # Logarithmic transformation
        df[var] = np.log(df[var])

    return df


def preprocess_data_after_eda(data: pd.DataFrame, num_var01: list, num_var: list, cat_bin_var: list, cat_var: list):
    """Preprocess data on results got in Exploratory Data Analysis"""

    df = update_null_values_(data)
    df = drop_useless_variables_(df, num_var01, num_var, cat_bin_var, cat_var)
    df = fix_day_price_error(df)
    df = drop_correlating_variables_(df, num_var01, num_var, cat_bin_var, cat_var)
    df = apply_logarithmic_transformation_(df, num_var)

    return df


# ======================================================================================
# Function for cross-validation pipeline
# ======================================================================================
QUANTITATIVE_PARAMS_FILE = 'model/{}_quantitative_params.json'
CATEGORICAL_PARAMS_FILE = 'model/{}_categorical_params.json'
VERSION = 1


def save_params(filename: str, data: dict):
    """Saving params to file in JSON"""

    with open(filename, 'w', encoding='utf-8') as file:
        return file.write(json.dumps(data, indent=4))


def save_model(clf, clf_name, prefix, version=VERSION):
    """Save model"""

    with open('model/{}_{}_{}_mdl.pkl'.format(prefix, version, clf_name.lower()), 'wb') as file:
        pickle.dump(clf, file)


def save_scaler(scl, prefix, version=VERSION):
    """Save scaler"""

    with open('model/{}_{}_sk.pkl'.format(prefix, version), 'wb') as file:
        pickle.dump(scl, file)


def load_params(filename: str):
    """Reading params from JSON stored in file"""

    with open(filename, 'r', encoding='utf-8') as file:
        return json.loads(file.read())


def load_model(clf_name, prefix, version=VERSION):
    """Load model"""

    with open('model/{}_{}_{}_mdl.pkl'.format(prefix, version, clf_name.lower()), 'rb') as file:
        return pickle.load(file)


def load_scaler(prefix, version=VERSION):
    """Load scaler"""

    with open('model/{}_{}_sk.pkl'.format(prefix, version), 'rb') as file:
        return pickle.load(file)


def process_outliers_for_cv(data: pd.DataFrame, num_var: list, cat_bin_var: list, num_var01: list, train=True,
                            prefix=''):
    """
    Process outliers on cross validation. Values defined on train sample are used on test sample
    """

    # In previous steps some variables were deleted, save only currently existing variables
    num_var = [nv for nv in num_var if nv in data.columns]
    num_var01 = [nv01 for nv01 in num_var01 if nv01 in data.columns]

    if train:
        params = {'percentile': {}}
        scaler = StandardScaler()
    else:
        params = load_params(QUANTITATIVE_PARAMS_FILE.format(prefix))
        scaler = load_scaler(prefix)

    # Preprocessing of quantitative variables (Q)
    for nv in data[num_var]:
        # New variable for saving outliers
        new_var_name = nv + '_out'
        data[new_var_name] = 0
        cat_bin_var.append(new_var_name)

        if train:
            dlimit = np.percentile(data[nv].values, 1)
            ulimit = np.percentile(data[nv].values, 99)
            params['percentile'][nv] = (dlimit, ulimit)
        else:
            dlimit = params['percentile'][nv][0]
            ulimit = params['percentile'][nv][1]

        # Set 1 if value is outlier 
        data[new_var_name] = data[nv].apply(lambda val: 1 if val < ulimit or val > dlimit else 0)

        # Lower and higher values equal to 1st and 99th percentile correspondingly
        data[nv] = data[nv].clip(lower=dlimit, upper=ulimit)

    # Preprocessing of quantitative variables with values in [0, 1] (Q01)
    for nv01 in data[num_var01]:
        # New variable for saving outliers
        new_var_name = nv01 + '_out'
        data[new_var_name] = 0
        cat_bin_var.append(new_var_name)

        if train:
            dlimit = np.percentile(data[nv01].values, 1)
            ulimit = np.percentile(data[nv01].values, 99)
            params['percentile'][nv01] = (dlimit, ulimit)
        else:
            dlimit = params['percentile'][nv01][0]
            ulimit = params['percentile'][nv01][1]

        # Set 1 if value is outlier 
        data[new_var_name] = data[nv01].apply(lambda val: 1 if val < ulimit or val > dlimit else 0)

        # Lower and higher values equal to 1st and 99th percentile correspondingly
        data[nv01] = data[nv01].clip(lower=dlimit, upper=ulimit)

    if train:
        save_params(QUANTITATIVE_PARAMS_FILE.format(prefix), params)
        data.loc[:, num_var] = scaler.fit_transform(data[num_var])
        save_scaler(scaler, prefix)
    else:
        data.loc[:, num_var] = scaler.transform(data[num_var])

    return data


def make_woe_encoding_for_cv(data: pd.DataFrame, cat_var: list, cat_bin_var: list, train=True, prefix=''):
    """
    Encode WoE: save coding from training sample and transfer them to test sample
    """

    # In previous steps some variables were deleted, save only currently existing variables
    cat_var = [cv for cv in cat_var if cv in data.columns]
    cat_bin_var = [cb for cb in cat_bin_var if cb in data.columns]

    if train:
        # Variable for storing train params
        # grouping - for values that is grouped
        # woe - for WoE values
        params = {
            'grouping': {},
            'woe': {}
        }

        # Grouping values that met in less than 0,5% case
        for cv in cat_var:
            params['grouping'][cv] = []
            cnt = data[cv].value_counts()
            for val, count in zip(cnt.index, cnt.values):
                if count / data.shape[0] <= 0.005:
                    params['grouping'][cv].append(val)
                    data.loc[data[cv] == val, cv] = 'NEW'

        bad_cntr = data[data.cntr_result == 1]
        good_cntr = data[data.cntr_result == 0]

        # WoE encoding
        for cv in cat_var:
            cnt = data[cv].value_counts()
            params['woe'][cv] = {}
            for val, count in zip(cnt.index, cnt.values):
                good_with_val = good_cntr[good_cntr[cv] == val].shape[0]
                bad_with_val = bad_cntr[bad_cntr[cv] == val].shape[0]

                p = good_with_val / good_cntr.shape[0]
                q = bad_with_val / bad_cntr.shape[0]
                woe = round(np.log(p / q), 3)

                params['woe'][cv][val] = woe
                data.loc[data[cv] == val, cv] = woe

        save_params(CATEGORICAL_PARAMS_FILE.format(prefix), params)
    else:
        params = load_params(CATEGORICAL_PARAMS_FILE.format(prefix))

        for cv in cat_var:
            # Grouping
            if params['grouping'][cv]:
                data[cv] = data[cv].replace(params['grouping'][cv], 'NEW')

            # WoE encofing
            data[cv] = data[cv].astype(str).map(params['woe'][cv])

            # If in test sample there is value that was not met in train sample
            if np.sum(data[cv].isnull()) > 0:
                # Encoding as for 'NEW' variable
                new_woe_code = params['woe'][cv].get('NEW', None)
                if new_woe_code:
                    # Changing unknown values on WoE code fore 'NEW' value
                    data[cv] = data[cv].fillna(new_woe_code)
                else:
                    data[cv] = data[cv].fillna(0)

    return data


def preprocess_data_for_cv(
        data: pd.DataFrame,
        num_var: list, cat_bin_var: list, cat_var: list,
        num_var01=['okpd_good_share_min', 'sup_cntr_avg_penalty_share', 'org_sim_price_share'],
        train=True, prefix=''):
    """
    Processing outliers, grouping rare values, WoE encoding on train sample.
    Transfering saved params to test sample.
    """

    data = process_outliers_for_cv(data, num_var, cat_bin_var, num_var01, train=train, prefix=prefix)
    data = make_woe_encoding_for_cv(data, cat_var, cat_bin_var, train=train, prefix=prefix)

    X = data.drop(['cntr_result'], axis=1).values
    y = data.cntr_result.values

    return X, y
