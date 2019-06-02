#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Library with functions for data analysis
"""

import math    
from collections import Counter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.feature_selection import SelectKBest, chi2, f_classif

GOOD_CNTR_CLR = '#5FFF92'
BAD_CNTR_CLR = '#FF665F'

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
    
def group_variables(data: pd.DataFrame):
    """
    Grouping variables in numeric with values between [0, 1] and other numeric,
    binary categorical and other categorical
    """
    
    num_var = [
        'sup_cntr_num', 
        'sup_running_cntr_num', 
        'sup_cntr_avg_price',
        'org_cntr_num',
        'org_running_cntr_num',
        'org_cntr_avg_price',
        'cntr_num_together',
        'okpd_cntr_num',
        'cntr_okpd_num',
        'plan_cntr_len',
        'day_price',
        'pmp',
        'price']

    num_var01 = [
        'sup_cntr_avg_penalty_share', 
        'sup_no_pnl_share', 
        'sup_1s_sev',
        'sup_1s_org_sev', 
        'sup_sim_price_share',
        'sup_good_cntr_share',
        'sup_fed_cntr_share',
        'sup_sub_cntr_share',
        'sup_mun_cntr_share',
        'org_1s_sev',
        'org_1s_sup_sev',
        'org_sim_price_share',
        'org_good_cntr_share',
        'org_fed_cntr_share',
        'org_sub_cntr_share',
        'org_mun_cntr_share',
        'okpd_good_share_min',
        'okpd_good_share_mean',
        'okpd_good_share_max']

    cat_var = [
        'sup_ter',
        'org_type',
        'org_ter',
        'sign_month',
        'sign_quarter',
        'purch_type',
        'cntr_lvl']

    cat_bin_var = (
        ['price_higher_pmp', 'price_too_low'] + 
        [clm for clm in data.columns if clm.startswith('okpd2_') or clm.startswith('socs')]
    )
    
    return num_var01, num_var, cat_bin_var, cat_var

def plot_outliers(variables: list, data: pd.DataFrame, lower_per=1, upper_per=99):
    """
    Plot graphs to check if variables have outliers 
    """
    
    num_of_rows = math.ceil(len(variables) / 4)
    fig = plt.figure(figsize=(20, num_of_rows * 5))
    df = data.copy()

    for idx, var in enumerate(variables):
        ax = fig.add_subplot(num_of_rows, 4, idx + 1)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        
        ulimit = np.percentile(df[var].values, upper_per)
        dlimit = np.percentile(df[var].values, lower_per)
        df[var] = df[var].clip(lower=dlimit, upper=ulimit)

        ax.scatter(range(data.shape[0]), np.sort(data[var]), label='Initial')
        ax.scatter(range(df.shape[0]), np.sort(df[var]), label='{}-{} percentile'.format(
            lower_per, upper_per))
        ax.legend()
        ax.set_title(var)
        ax.set_xticklabels(['']*len(labels))
        
def plot_histograms(variables: list, data: pd.DataFrame, lower_per=1, upper_per=99):
    """Histograms of variable in defined boarder"""
    
    num_of_rows = math.ceil(len(variables) / 4)
    fig = plt.figure(figsize=(20, num_of_rows * 5))
    df = data.copy()

    for idx, var in enumerate(variables):
        ax = fig.add_subplot(num_of_rows, 4, idx + 1)

        ulimit = np.percentile(data[var].values, 99)
        dlimit = np.percentile(data[var].values, 1)
        df.loc[df[var] > ulimit, var] = ulimit
        df.loc[df[var] < dlimit, var] = dlimit

        df[var].hist(ax=ax)
        ax.set_title(var)
        
def rate_feature_importance(
    X_values: np.ndarray, y_values: np.ndarray, criteria: list, criteria_names: list, columns: list, alias=''
):
    """Assess feature importance by criteria"""
    
    result_dict = {}
    result_list = []
    
    for criterium in criteria:
        test = SelectKBest(score_func=criterium, k='all')
        fit = test.fit(X_values, y_values)
        res = sorted(
            dict(zip(columns, [s for s in list(fit.scores_)])).items(),
            key=lambda a: a[1],
            reverse=True
        )
        
        func_name = str(criterium).split()[1]
        dict_key = '{}_{}'.format(alias, func_name) if alias else func_name
        
        result_dict[dict_key] = [r[0] for r in res]
        result_list.append(res)
   
    res_df = pd.DataFrame(result_dict)
    for idx, cr_name in enumerate(criteria_names):
        res_df[cr_name] = [int(i[idx + 1]) for i in result_list[0]]
    
    res_df.columns = ['var_name'] + criteria_names
    
    return res_df, result_list

def calculate_information_value(cat_variables: list, data: pd.DataFrame):
    """
    Calculation of IV. Link to example: 
    https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb
    """
    
    # Grouping rare values (met in less then 0,5% cases)
    df = data.copy()
    for cv in cat_variables:
        cnt = data[cv].value_counts()
        for val, count in zip(cnt.index, cnt.values):
            if count / data.shape[0] <= 0.005:
                df.loc[df[cv] == val, cv] = 'NEW'

    # Calculation of Information Value (IV)
    good_cntr = df.loc[data.cntr_result == 0]
    bad_cntr = df.loc[data.cntr_result == 1]

    good_num = df.loc[data.cntr_result == 0].shape[0]
    bad_num = df.loc[data.cntr_result == 1].shape[0]
    res = {}

    for cv in cat_variables:
        res[cv] = []
        for val in set(df[cv].values):
            good_with_val = good_cntr[good_cntr[cv] == val].shape[0]
            bad_with_val = bad_cntr[(bad_cntr[cv] == val)].shape[0]
            p = good_with_val / good_num
            q = bad_with_val / bad_num
            w = np.log(p / q)
            res[cv].append((p - q) * w)

    for key, value in res.items():
        print('{}: {:.3f}'.format(key, sum(res[key])))
         
def cntr_distrib_over_cat_var(data: pd.DataFrame, cat_var: str, fig_width=20, percent=True):
    """
    Visualization of distribution of good and bad contracts over categorical variable
    """
    
    df = data.copy()
    total, bad, good = Counter(), Counter(), Counter()
    cat_values, total_count, bad_count, good_count = [], [], [], []

    for row in df[[cat_var, 'cntr_result']].itertuples():
        cat_var_val = getattr(row, cat_var)
        cntr_res = row.cntr_result
        
        total[cat_var_val] += 1
        
        if cntr_res == 1:
            bad[cat_var_val] += 1
        else:
            good[cat_var_val] += 1

    for val in np.unique(df[cat_var]):
        cat_values.append(val)
        total_count.append(total.get(val))
        bad_count.append(bad.get(val, 0))
        good_count.append(good.get(val, 0))

    df = pd.DataFrame({
        cat_var: cat_values, 
        'bad': bad_count, 
        'good': good_count, 
        'total': total_count
    }) 
    
    r = range(len(cat_values))
    fig = plt.figure(figsize=(fig_width,5))
    ax = plt.subplot(111)
    plt.xticks(r, cat_values, rotation='vertical')
    plt.xlabel(cat_var)
    
    if percent:
        df['bad_prop'] = df['bad'] / df['total']
        df['good_prop'] = df['good'] / df['total']

        ax.bar(r, df['bad_prop'], color=BAD_CNTR_CLR, edgecolor='white', label='Bad contracts')
        ax.bar(r, df['good_prop'], bottom=df['bad_prop'], color=GOOD_CNTR_CLR, edgecolor='white', label='Good contracts')
        plt.ylabel("Share")
        plt.title("Share of good and bad contracts over {}".format(cat_var))
    else:
        ax.bar(r, df['bad'], color=BAD_CNTR_CLR, edgecolor='white', label='Bad contracts')
        ax.bar(r, df['good'], bottom=df['bad'], color=GOOD_CNTR_CLR, edgecolor='white', label='Good contracts')
        plt.ylabel("Number")
        plt.title("Number of good and bad contracts over {}".format(cat_var))
    
    ax.legend(loc='upper center')
    plt.show()