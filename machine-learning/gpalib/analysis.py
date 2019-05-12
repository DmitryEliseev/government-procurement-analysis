#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Library with functions for data analysis
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn


def okpd_analysis(data: pd.DataFrame, per1=1, per2=99):
    """Analysis of contracts with multiple OKPD"""

    df = (data.groupby('cntrID')['okpd']
          .agg('count')
          .reset_index()
          .sort_values(by='okpd', ascending=False))

    df_okpds = df[df.okpd > 1]
    perc_value1 = np.percentile(df_okpds.okpd, per1)
    perc_value2 = np.percentile(df_okpds.okpd, per2)

    print('Доля контрактов с несколькими ОКПД: {:.2f}'.format(df_okpds.shape[0] / df.shape[0]))
    print('{} перценитиль: {}'.format(per1, perc_value1))
    print('{} перценитиль: {}'.format(per2, perc_value2))

    sn.distplot(df_okpds.okpd, ax=plt.figure(figsize=(20, 5)).gca())
    plt.title('Распределение кол-ва ОКПД на контракт')
    plt.show()

    sn.distplot(
        df_okpds[(df_okpds.okpd >= perc_value1) & (df_okpds.okpd <= perc_value2)].okpd,
        ax=plt.figure(figsize=(20, 5)).gca())
    plt.title('Распределение кол-ва ОКПД на контракт в рамках [{}, {}] перцентилей'.format(per1, per2))
    plt.show()

    return df

def okpd_unique_count(data: pd.DataFrame):
    """Number of unique OKPD depending on OKPD length"""
    
    df = data.copy()
    max_len_okpd = len(str(max(data.okpd, key=lambda a: len(str(a)))))
    res = []
    
    for okpd_syms in range(1, max_len_okpd + 1):
        okpd_column_name = 'okpd{}'.format(okpd_syms)
        df[okpd_column_name] = df.okpd.apply(lambda a: str(a)[:okpd_syms])
        res.append([okpd_syms, len(np.unique(df[okpd_column_name]))])
    
    return pd.DataFrame(res, columns=['okpd_sym', 'count'])

def contracts_with_many_okpd(data: pd.DataFrame, debug=True):
    """Get list of contracts with several OKPD"""
    
    cntrID_with_many_okpd = []
    for key, value in data.cntrID.value_counts().items():
        if value == 1:
            break
        cntrID_with_many_okpd.append(key)
        
    cntr_many_okpd = data[data.cntrID.isin(cntrID_with_many_okpd)]
    
    if debug:
        print('Number of unique contracts: {}'.format(len(cntrID_with_many_okpd)))
        print('Number of observations in sample: {}'.format(cntr_many_okpd.shape[0]))

    return cntrID_with_many_okpd, cntr_many_okpd

def good_bad_cntr_many_okpd(data: pd.DataFrame):
    """Proportion of good and bad contracts among contracts with several OKPD"""
    
    cntrIDs, data_many_okpd = contracts_with_many_okpd(data)
    
    data_many_okpd = data_many_okpd.drop_duplicates('cntrID')
    print(data_many_okpd.cntr_result.value_counts())
    
    sn.countplot(x='cntr_result', data=data_many_okpd)