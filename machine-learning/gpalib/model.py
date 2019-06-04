#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Library with functions for model analytics
"""

import time
import itertools
from collections import Counter

import pandas as pd
import numpy as np

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.calibration import calibration_curve

from gpalib.preprocessing import preprocess_data_for_cv
from gpalib.analysis import group_variables

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 42


class Classifier():
    """Class for keeping classifier"""

    def __init__(self, clf, name, short_name, model_code):
        self.clf = clf
        self.name = name
        self.short_name = short_name
        self.model_code = model_code

    y_train_real = np.array([])
    y_test_real = np.array([])

    y_train_pred = np.array([])
    y_test_pred = np.array([])

    y_train_pred_proba = np.zeros((0, 2))
    y_test_pred_proba = np.zeros((0, 2))


def real_and_predicted_correlation(y_test_real, y_test_pred, y_test_pred_proba):
    """Matching between real and predicted values"""

    return (
        pd.DataFrame({'result': y_test_real})
            .join(pd.DataFrame({'proba_pred': y_test_pred_proba[:, 1]})
                  .join(pd.DataFrame({'pred': y_test_pred})))
    )


def plot_roc_curve(clf):
    """Plotting ROC"""

    fpr, tpr, threshold = roc_curve(clf.y_test_real, clf.y_test_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC')
    plt.show()


def assess_probability_power(clfs, figsize=(20, 10)):
    """Visualization of probabilistic power of classifier"""

    fig = plt.figure(figsize=figsize)
    clf_num = len(clfs)
    res = {}

    if clf_num > 4:
        raise Warning('This function is recommended to use for 1-4 models')

    for idx, clf in enumerate(clfs):
        ax1 = fig.add_subplot(2, clf_num, idx + 1)
        ax1.plot([0, 1], [0, 1], "k:", label="Ideal classifier")
        ax2 = fig.add_subplot(2, clf_num, idx + 1 + clf_num)

        fraction_of_positives, mean_predicted_value = calibration_curve(
            clf.y_test_real, clf.y_test_pred_proba[:, 1], n_bins=10)

        ax1.plot(
            mean_predicted_value, fraction_of_positives, "s-",
            label='%s (log loss: %1.3f)' % (clf.short_name, log_loss(clf.y_test_real, clf.y_test_pred_proba))
        )
        ax2.hist(clf.y_test_pred_proba[:, 1], range=(0, 1), bins=10, histtype="step", lw=2)

        ax1.set_ylabel('Share of observations of bad classes')
        ax1.set_xlabel('Average predicted probability')
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc='lower right')
        ax1.set_title(clf.short_name)

        ax2.set_ylabel('Number of observations')
        ax2.set_xlabel('Average predicted probability')


def plot_training_time(classifiers, figsize=(10, 5)):
    """Plot time of model training"""

    fig = plt.figure(figsize=figsize)

    y_pos = range(len(classifiers))
    train_time = [clf.train_time for clf in classifiers]

    plt.bar(y_pos, train_time, alpha=0.5)
    plt.xticks(y_pos, [clf.short_name for clf in classifiers])
    plt.title('Time of training')
    plt.ylim(0, int(max(train_time) * 1.25))

    for idx, tt in enumerate(train_time):
        plt.text(idx, tt + 10, str(tt), ha='center')

    plt.show()


def feature_importance(classifiers: list, data: pd.DataFrame):
    """Feature importance analysis"""

    result = {}

    for clf in classifiers:
        feature_importance = None

        # For logistic regression (coef_)
        if clf.model_code == 'LR':
            feature_importance = sorted(
                list(zip(data.columns[1:-1], clf.clf.coef_[0])),
                key=lambda a: np.abs(a[1]),
                reverse=True
            )
        # Fot other models (feature_importances_)
        else:
            feature_importance = sorted(
                list(zip(data.columns[1:-1], clf.clf.feature_importances_)),
                key=lambda a: a[1],
                reverse=True
            )

        result[clf.short_name] = ['{}: {:.2f}'.format(elem[0], elem[1]) for elem in feature_importance]

    return pd.DataFrame(result)


def plot_confusion_matrixes(
        classifiers, normalize=False, title='Confusion matrix',
        figsize=(20, 10), cmap=plt.cm.Blues
):
    """Plotting confusion matrix """

    fig = plt.figure(figsize=figsize)
    num_of_clf = len(classifiers)
    tick_marks = np.arange(2)

    for idx, clf in enumerate(classifiers):
        ax = fig.add_subplot(1, num_of_clf, idx + 1)

        cm = confusion_matrix(clf.y_test_real, clf.y_test_pred)
        if clf.model_code.startswith('NN'):
            classes = [0, 1]
        else:
            classes = clf.clf.classes_

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax.imshow(cm, interpolation='nearest', cmap=cmap)

        ax.set_title(title + ' ' + clf.short_name)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True values')
        ax.set_xlabel('Predicted values')


def custom_classification_report(classifiers):
    """
    Custom classification_report for several models
    """

    scores = []
    for clf in classifiers:
        scores.append([
            precision_score(clf.y_test_real, clf.y_test_pred, pos_label=0),
            precision_score(clf.y_test_real, clf.y_test_pred, pos_label=1),
            recall_score(clf.y_test_real, clf.y_test_pred, pos_label=0),
            recall_score(clf.y_test_real, clf.y_test_pred, pos_label=1),
            f1_score(clf.y_test_real, clf.y_test_pred),
            accuracy_score(clf.y_test_real, clf.y_test_pred)]
        )

    return pd.DataFrame(
        np.transpose(np.array(scores)),
        index=['precision_0', 'precision_1', 'recall_0', 'recall_1', 'f_score', 'accuracy'],
        columns=[clf.short_name for clf in classifiers]).round(3)


def plot_dependencies(classifiers, figsize=(20, 10)):
    """
    Custom visualization of probabilistic power of classifier
    """

    fig = plt.figure(figsize=figsize)
    num_of_clf = len(classifiers)

    for idx, clf in enumerate(classifiers):
        ax = fig.add_subplot(230 + idx + 1)

        counter0, counter1 = Counter(), Counter()

        if not hasattr(clf, 'res'):
            clf.res = real_and_predicted_correlation(
                clf.y_test_real,
                clf.y_test_pred,
                clf.y_test_pred_proba
            )

        for index, row in clf.res.iterrows():
            proba = round(row['proba_pred'], 2)

            if row['result']:
                counter1[proba] += 1
            else:
                counter0[proba] += 1

        corr_df0 = pd.DataFrame.from_dict(counter0, orient='index').reset_index().sort_values(['index'])
        corr_df1 = pd.DataFrame.from_dict(counter1, orient='index').reset_index().sort_values(['index'])

        ax.plot(corr_df0['index'], corr_df0[0], label='Good contracts')
        ax.plot(corr_df1['index'], corr_df1[0], label='Bad contracts')

        ax.legend()
        ax.set_xlabel('Predicted probability of contract being bad')
        ax.set_ylabel('Real number of contracts')
        ax.set_title(clf.short_name)


def visualize_clfs_quality(classifiers):
    """All basic charts to access classifier quality"""

    # Confusion matrix
    plot_confusion_matrixes(classifiers)

    # Probabilistic power of model
    plot_dependencies(classifiers)

    # Alternative way of visualizing probalistic quality of model
    assess_probability_power(classifiers)


def transform_cros_val_scores(scores: dict):
    """Transformation of output of next function to neat view"""

    columns = [
        'model', 'time',
        'tst_acc', 'tst_acc_std',
        'tst_auc', 'tst_auc_std',
        'tst_ll', 'tst_ll_std',
        'tr_acc', 'tr_acc_std',
        'tr_auc', 'tr_auc_std',
        'tr_ll', 'tr_ll_std'
    ]

    res = {}
    for clmn in columns:
        res[clmn] = []

    for key, value in scores.items():
        res['model'].append(key)
        res['time'].append(np.mean(value['fit_time']))

        res['tr_acc'].append(np.mean(value['train_accuracy']))
        res['tr_acc_std'].append(np.std(value['train_accuracy']))

        res['tr_auc'].append(np.mean(value['train_roc_auc']))
        res['tr_auc_std'].append(np.std(value['train_roc_auc']))

        res['tr_ll'].append(np.mean(value['train_neg_log_loss']))
        res['tr_ll_std'].append(np.std(value['train_neg_log_loss']))

        res['tst_acc'].append(np.mean(value['test_accuracy']))
        res['tst_acc_std'].append(np.std(value['test_accuracy']))

        res['tst_auc'].append(np.mean(value['test_roc_auc']))
        res['tst_auc_std'].append(np.std(value['test_roc_auc']))

        res['tst_ll'].append(np.mean(value['test_neg_log_loss']))
        res['tst_ll_std'].append(np.std(value['test_neg_log_loss']))

    return pd.DataFrame(res)


def cross_validate(clf, data, scoring, cv=3, prefix='', silent=False):
    """Function for manual cross validation"""

    scores = {}
    aliases = ('train_', 'test_')
    scores['fit_time'] = []

    for alias in aliases:
        for metric in scoring:
            scores[alias + metric] = []

    X, y = data.iloc[:, :-1].values, data.cntr_result.values

    skf = StratifiedKFold(n_splits=cv, random_state=RANDOM_SEED)
    for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]

        num_var01, num_var, cat_bin_var, cat_var = group_variables(data)

        X_train, y_train = preprocess_data_for_cv(data_train, num_var, cat_bin_var, cat_var, train=True, prefix=prefix)
        X_test, y_test = preprocess_data_for_cv(data_test, num_var, cat_bin_var, cat_var, train=False, prefix=prefix)

        start_time = time.time()
        clf.clf.fit(X_train, y_train)
        clf.train_time = int(time.time() - start_time)
        scores['fit_time'].append(clf.train_time)

        if not silent:
            print('{}: fold {}'.format(clf.short_name, idx + 1))

        clf.y_train_real = y_train
        clf.y_test_real = y_test
        clf.y_train_pred = clf.clf.predict(X_train)
        clf.y_test_pred = clf.clf.predict(X_test)
        clf.y_train_pred_proba = clf.clf.predict_proba(X_train)
        clf.y_test_pred_proba = clf.clf.predict_proba(X_test)

        # Matching between real and predicted values
        clf.res = real_and_predicted_correlation(
            clf.y_test_real,
            clf.y_test_pred,
            clf.y_test_pred_proba
        )

        for alias in aliases:
            if alias == 'train_':
                y_true = y_train
                y_pred = clf.y_train_pred
                y_pred_proba = clf.y_train_pred_proba
            else:
                y_true = y_test
                y_pred = clf.y_test_pred
                y_pred_proba = clf.y_test_pred_proba

            for metric in scoring:
                if metric == 'accuracy':
                    score = accuracy_score(y_true, y_pred)
                elif metric == 'roc_auc':
                    score = roc_auc_score(y_true, y_pred_proba[:, 1])
                elif metric == 'neg_log_loss':
                    score = -log_loss(y_true, y_pred_proba[:, 1])

                scores[alias + metric].append(score)
    return scores


def validate_model(clf, train_data, valid_data, scoring, silent=False, prefix=''):
    """Function for training one model"""

    scores = {}
    aliases = ('train_', 'test_')
    scores['fit_time'] = []

    for alias in aliases:
        for metric in scoring:
            scores[alias + metric] = []

    num_var01, num_var, cat_bin_var, cat_var = group_variables(train_data)
    X_train, y_train = preprocess_data_for_cv(train_data, num_var, cat_bin_var, cat_var, train=True, prefix=prefix)
    X_val, y_val = preprocess_data_for_cv(valid_data, num_var, cat_bin_var, cat_var, train=False, prefix=prefix)

    start_time = time.time()
    clf.clf.fit(X_train, y_train)
    clf.train_time = int(time.time() - start_time)
    scores['fit_time'].append(clf.train_time)

    if not silent:
        print('Model %s was trained' % clf.short_name)

    clf.y_train_real = y_train
    clf.y_test_real = y_val
    clf.y_train_pred = clf.clf.predict(X_train)
    clf.y_test_pred = clf.clf.predict(X_val)
    clf.y_train_pred_proba = clf.clf.predict_proba(X_train)
    clf.y_test_pred_proba = clf.clf.predict_proba(X_val)

    # Matching between real and predicted values
    clf.res = real_and_predicted_correlation(
        clf.y_test_real,
        clf.y_test_pred,
        clf.y_test_pred_proba
    )

    for alias in aliases:
        if alias == 'train_':
            y_true = y_train
            y_pred = clf.y_train_pred
            y_pred_proba = clf.y_train_pred_proba
        else:
            y_true = y_val
            y_pred = clf.y_test_pred
            y_pred_proba = clf.y_test_pred_proba

        for metric in scoring:
            if metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'roc_auc':
                score = roc_auc_score(y_true, y_pred_proba[:, 1])
            elif metric == 'neg_log_loss':
                score = -log_loss(y_true, y_pred_proba[:, 1])

            scores[alias + metric].append(score)
    return scores


def hyperopt_train_test(train_data: pd.DataFrame, params: dict, history_storage: dict, cv=2, prefix='hp'):
    """Function for parameter tuning with Hyperopt"""

    clf_type = params['type']
    del params['type']

    if clf_type == 'LogReg':
        clf = LogisticRegression(**params)
    elif clf_type == 'RandForest':
        clf = RandomForestClassifier(**params)
    elif clf_type == 'XGBoost':
        clf = XGBClassifier(**params)

    scores = cross_validate(
        Classifier(clf, '', clf_type, ''),
        train_data,
        ['neg_log_loss'],
        cv=cv,
        silent=True,
        prefix=prefix)

    score = np.mean(scores['test_neg_log_loss'])

    # Save important params and score
    history_storage['model'].append(clf_type)
    history_storage['params'].append(params)
    history_storage['score'].append(score)

    return score
