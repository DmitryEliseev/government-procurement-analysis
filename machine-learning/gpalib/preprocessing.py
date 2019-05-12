#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Library with functions for data preprocessing
"""

import numpy as np
import pandas as pd

# ======================================================================================
# Data aggregation after shortening OKPD variable
# ======================================================================================
    
def shorten_okpd_(data: pd.DataFrame, okpd_sym=2):
    """Creation of column with shortened OKPD"""
    
    df = data.copy()
    
    okpd_column_name = 'okpd{}'.format(okpd_sym)
    df[okpd_column_name] = df.okpd.apply(lambda a: str(a)[:okpd_sym])
    unique_okpd = np.unique(df[okpd_column_name])
    
    return df, okpd_column_name, unique_okpd

def agg_stats_for_okpd_(data: pd.DataFrame, okpd_column: str, unique_okpds: list):
    """
    Aggregation for columns `okpd_cntr_num` and `okpd_good_cntr_num` for shortened OKPD
    """
    
    df = data.copy()
    
    df_ = df.drop_duplicates('okpd')
    agg_df = (
        df_[['okpd_cntr_num', 'okpd_good_cntr_num', okpd_column]]
        .groupby(okpd_column)[['okpd_cntr_num', 'okpd_good_cntr_num']]
        .agg('sum')
    )
    
    for row in agg_df.itertuples():
        df.loc[df[okpd_column] == row.Index, 'okpd_cntr_num'] = row.okpd_cntr_num
        df.loc[df[okpd_column] == row.Index, 'okpd_good_cntr_num'] = row.okpd_good_cntr_num
        
    return df

def agg_stats_for_sup_(data: pd.DataFrame, okpd_column: str, unique_okpds: list):
    """
    Aggregation for column `sup_okpd_cntr_num` for shortened OKPD
    """
    
    df = data.copy()

    df_ = df.drop_duplicates(['supID', 'okpd'])
    agg_df = (
        df_[['sup_okpd_cntr_num', 'okpd2', 'supID']]
        .groupby(['supID', 'okpd2'])['sup_okpd_cntr_num']
        .agg('sum')
    )
    
    for row in agg_df.iteritems():
        df.loc[(df.supID == row[0][0]) & (df[okpd_column] == row[0][1]), 'sup_okpd_cntr_num'] =  row[1]
    
    return df

def aggregate_data_by_shortened_okpd(data: pd.DataFrame, okpd_sym=2, debug=True):
    df, okpd_column_name, unique_okpd = shorten_okpd_(data, okpd_sym)
    if debug:
        print('New column `{}` is added'.format(okpd_column_name))
    
    df = agg_stats_for_okpd_(df, okpd_column_name, unique_okpd)
    if debug:
        print('Data for columns `okpd_cntr_num` and `okpd_good_cntr_num` is aggregated')

    df = agg_stats_for_sup_(df, okpd_column_name, unique_okpd)
    if debug:
        print('Data for column `sup_okpd_cntr_num` is aggregated')
    
    return df

def get_dummies_for_okpd_vars(data: pd.DataFrame, okpd_column_name: str, debug=True):
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
    
    return pd.concat([df, okpd_dummies, sup_okpd_cntr_share_dummies], axis=1)

def create_new_var_based_on_okpd(data: pd.DataFrame, okpd_column_name, debug=True):
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
    cntrIDs, cntr_many_okpd = gpalib.analysis.contracts_with_many_okpd(df, debug=False)
    cntrID_cv = cntr_many_okpd.cntrID.value_counts()
    df['okpd_num'] = df.cntrID.apply(lambda cntr_id: cntrID_cv.get(cntr_id, 1))
    
    return df

def flatten_agg_okpds(data: pd.DataFrame, okpd_column_name, debug=True):
    """Aggregation of data about many OKPD of contract in one observation"""
    
    df = data.copy()
    
    # cntrIDs with several OKPD
    cntrIDs, _ = gpalib.analysis.contracts_with_many_okpd(df, debug=False)
    
    # dummy columns with OKPD
    okpd_columns = [col for col in df.columns if okpd_column_name + '_' in col]
    
    # OKPD aggregation in `res` variable
    for idx, cntrID in enumerate(cntrIDs):
        okpd_agg = np.sum(
            df.loc[lambda d: d.cntrID == cntrID, okpd_columns[0]:okpd_columns[-1]], 
            axis=0).values
        
        if not idx:
            res = okpd_agg
        else:
            res = np.vstack((res, okpd_agg)) 
    
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
    df_many_okpd.loc[lambda df: df.cntrID.isin(cntrIDs), okpd_columns[0]:okpd_columns[-1]] = res
    
    # Concatenation of two dataframes
    df = pd.concat([df_one_okpd, df_many_okpd], sort=False)
    
    return df.sample(frac=1).reset_index(drop=True)

def transform_okpd_good_share(data: pd.DataFrame, data_with_dup: pd.DataFrame, okpd_column_name: str, debug=True):
    """
    Transformation of `okpd_good_cntr_share` for cases, when contract included several OKPD.
    Adding min, mean and max share over presented OKPD in contract.
    """
    
    df = data.copy()
    
    cntrIDs, _ = gpalib.analysis.contracts_with_many_okpd(data_with_dup, debug=False)
    
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
    
    return df