#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Library with functions for data preprocessing
"""

import numpy as np
import pandas as pd

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
        print('Data for column `sup_okpd_cntr_num` is aggregated')
    
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
        print('`socs_` variables were updated')
    
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