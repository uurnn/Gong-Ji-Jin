## coding:utf-8
import datetime

import os
import random
import time
import warnings


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost
import seaborn as sns
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from lianyhaii.model import tpr_weight_funtion

warnings.filterwarnings('ignore')


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def deal_time(train, test):
    # tmp_df = pd.concat([train,test],axis=0,ignore_index=True)
    for df in [train, test]:
        long_time_mask = df['CSNY'].astype(str).str.len() == 12
        df['time'] = 0
        df.loc[long_time_mask, 'time'] = pd.to_datetime(df.loc[long_time_mask, 'CSNY'], unit='ms')
        df.loc[~long_time_mask, 'time'] = pd.to_datetime(df.loc[~long_time_mask, 'CSNY'], unit='s')

        df['time'] = pd.to_datetime(df.loc[:, 'time'].copy())

def deal_dkll(train, test):
    ##只能存在四种利率：2.708（3.25），2.979（3.575），2.292，2.521
    for df in [train, test]:
        df['DKLL'] = np.round(df['DKLL'], 3)
        year5_M_rate_mask = df['DKLL'].isin([3.25, 3.025])
        year1_M_rate_mask = df['DKLL'].isin([2.750])
        year5_M2_rate_mask = df['DKLL'].isin([3.575])
        df.loc[year5_M_rate_mask, 'DKLL'] = 2.708
        df.loc[year1_M_rate_mask, 'DKLL'] = 2.292
        df.loc[year5_M2_rate_mask, 'DKLL'] = 2.979

        double_house_mask = df['DKLL'].isin([2.979, 2.5212])
        df['DKLL_check'] = (year5_M_rate_mask | year5_M2_rate_mask | year1_M_rate_mask).astype(int)
        df['double_house'] = (double_house_mask).astype(int)

    return ['DKLL_check', 'double_house']

def load_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    if drop_zhiwu:
        train = train[train['ZHIWU'] == 0]
        test = test[test['ZHIWU'] == 0]

    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    return train, test

def load_params(seed):
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_jobs': -1,
        'learning_rate': 0.01,
        'num_leaves': 2**8,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 3000,
        'max_bin': 255,
        'verbose': -1,
        # 'min_child_samples': 20,
        'seed': 0,
        'early_stopping_rounds': 100,
    }

    xgb_params = {
        'objective': 'binary:logistic',
        # 'objective':'binary:logitraw',
        # 'objective':'binary:hinge',
        # 'booster': 'gbtree',
        # 'eval_metric': 'mae',
        'eval_metric': 'auc',
        # 'num_boost_round':2000,
        # 'min_child_weight': 10,
        'tree_method': 'hist',
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.02,
        'seed': seed,
        'nthread': -1,
        # 'silent': True,
    }

    cat_params = {'learning_rate': 0.02,
                  'depth': 8,
                  'l2_leaf_reg': 10,
                  'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter',
                  'od_wait': 50, 'random_seed': seed,
                  'allow_writing_files': False}

    lr_params = {
        'random_state': seed,
        'C': 1,
        'max_iter': 1000,
        'n_jobs': -1,

    }

    model_params = {
        'lgb': lgb_params,
        # 'xgb': xgb_params,
        # 'cat':cat_params,
        # 'lr':lr_params,
    }

    return model_params


class make_test():
    def __init__(self, tr_df, tt_df, base_features, new_features, m_score, label):
        self.train = tr_df
        self.test = tt_df
        self.base_features = base_features
        self.new_features = new_features
        self.m_score = m_score
        self.label = label
        self.features = base_features + new_features
        self.predictions = None
        self.model = []

    def init_CV(self, seed, n_split=5, shuffle=True, CV_type='skFold'):
        self.CV_type = None
        if CV_type == 'skFold':
            self.CV_type = StratifiedKFold(n_splits=n_split, shuffle=shuffle, random_state=seed)
        if CV_type == 'kFold':
            self.CV_type = KFold(n_splits=n_split, shuffle=shuffle, random_state=seed)
        if CV_type == 'Noval':
            pass

    def __check_diff_score(self, oof_predictions):
        auc_score = roc_auc_score(y_true=self.train[self.label], y_score=oof_predictions)
        tpr_score = tpr_weight_funtion(y_true=self.train[self.label], y_predict=oof_predictions)
        # acc_score = accuracy_score(y_true=self.train[self.label], y_pred=oof_predictions > 0.5)
        print('global auc :', auc_score)
        print('global tpr :', tpr_score)
        # print('global acc is :',acc_score )
        print('=' * 10 + 'different with previous version' + '=' * 10)
        print('diff of auc :', np.round(auc_score - self.m_score[-1][0], 5))
        print('diff of tpr :', np.round(tpr_score - self.m_score[-1][1], 5))
        # print('diff of acc :',np.round(acc_score - self.m_score[-1][1],5))
        self.m_score.append([auc_score, tpr_score])

    def lgb_test(self, lgb_params, cv_score=False):

        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))

        for n, (trn, val) in enumerate(self.CV_type.split(self.train, self.train[self.label])):
            trn_X, trn_y = self.train.loc[trn, self.features], self.train.loc[trn, self.label]
            val_X, val_y = self.train.loc[val, self.features], self.train.loc[val, self.label]

            trn_data = lgb.Dataset(trn_X, label=trn_y)
            val_data = lgb.Dataset(val_X, label=val_y)

            estimator = lgb.train(lgb_params,
                                  trn_data,
                                  valid_sets=[trn_data, val_data],
                                  # feval=lgb_f1_score,
                                  # feval=tpr_weight_3_cunstom,
                                  verbose_eval=-1)

            oof_predictions[val] = estimator.predict(val_X,)
            self.model.append(estimator)

            cv_score_list.append(roc_auc_score(y_true=val_y, y_score=oof_predictions[val]))
            tt_predicts += estimator.predict(self.test[self.features]) / self.CV_type.n_splits
        print(f"training CV oof mean : {np.round(np.mean(cv_score_list), 5)}")

        self.__check_diff_score(oof_predictions)
        self.predictions = tt_predicts
        if cv_score:
            return oof_predictions, tt_predicts, cv_score_list
        else:
            return oof_predictions, tt_predicts

    def submit(self):
        today = time.strftime("%Y-%m-%d", time.localtime())[5:]
        self.test[self.label] = self.predictions
        sub_train = self.test[['id', self.label]].copy()
        sub_test = pd.read_csv('./data/submit.csv')
        sub = sub_test[['id']].merge(sub_train, on='id', how='left')
        plt.figure(figsize=(20, 10))
        sns.distplot(sub['label'], bins=100)
        plt.show()
        sub.fillna(0, inplace=True)
        score = str(np.round(self.m_score[-1][0], 4)) + "_" + str(np.round(self.m_score[-1][1], 4))
        sub.to_csv(f'sub_{today}_{score}.csv', index=False)

def encode_frq(df1,df2,cols):
    add_features = []
    for col in cols:
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=False, normalize=False).to_dict()
        vc[-1] = -1
        nm = col + '_FrqEnc'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm, ', ', end='\n')
        add_features.append(nm)
    return add_features


if __name__ == '__main__':

    seed = 0
    seed_everything(seed)
    drop_zhiwu = True

    label = 'label'
    ID = 'id'

    train, test = load_data()

    def add_person_features(dl_time=True):
        global train, test

        train['ZHIYE'] = train['ZHIYE'].map(lambda x: 90 if x in [11, 21] else x)
        test['ZHIYE'] = test['ZHIYE'].map(lambda x: 90 if x in [11, 21] else x)
        train['ZHICHEN'] = train['ZHICHEN'].map(lambda x: 999 if x in [99] else x).map(lambda x: 1 if x == 999 else 0)
        test['ZHICHEN'] = test['ZHICHEN'].map(lambda x: 999 if x in [99] else x).map(lambda x: 1 if x == 999 else 0)
        new_feats = ['XINGBIE', 'ZHIYE', 'ZHICHEN']
        ## 处理时间变量
        if dl_time:
            deal_time(train, test)
            train['age'] = 2020 - train['time'].dt.year
            test['age'] = 2020 - test['time'].dt.year

            train['time_tonow'] = (datetime.datetime.now() - train['time']).dt.days
            test['time_tonow'] = (datetime.datetime.now() - test['time']).dt.days


            new_feats += ['time_tonow', 'age']

        return new_feats

    def add_dw_features(test_dw_feats=True):
        global train, test


        new_feats = []
        for col in ['DWJJLX', 'DWSSHY']:
            train[col+"_cate"] = train[col].astype('category')
            test[col+"_cate"] = test[col].astype('category')
            new_feats.append([col+"_cate"])
        tt_new_features = [x for item in new_feats for x in item]
        if test_dw_feats:
            new_feats.append(tt_new_features)
            return new_feats
        else:
            return tt_new_features

    def add_zh_features(clean_outlier=True, test_zh_feats=True):

        global train, test
        zh_num_feats = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'DWYJCE']
        zh_cat_feats = ['GRZHZT']
        new_feats = []

        ## clean outlier
        if clean_outlier:
            train[zh_num_feats] = train[zh_num_feats].apply(lambda x: x - 237)
            test[zh_num_feats] = test[zh_num_feats].apply(lambda x: x - 237)

        ## deal with cat encode
        train['GRZHZT_cate'] = train['GRZHZT'].astype('category')
        test['GRZHZT_cate'] = test['GRZHZT'].astype('category')
        new_feats.append(['GRZHZT_cate'])

        ##业务特征
        ye_features = []
        train['DWJCBL'] = train['DWYJCE'] / (train['GRJCJS'])
        test['DWJCBL'] = test['DWYJCE'] / (test['GRJCJS'])

        ye_features.append('DWJCBL')
        new_feats.append(ye_features)
        ff = encode_frq(train, test, cols=ye_features)
        new_feats.append(ff)

        ##对数值特征进行cat化
        ff = encode_frq(train, test, cols=zh_num_feats)
        new_feats.append(ff)

        train['rate'] = np.round(train['GRZHDNGJYE'] / (train['DWYJCE'] * 2), 4)
        test['rate'] = np.round(test['GRZHDNGJYE'] / (test['DWYJCE'] * 2), 4)

        ff = encode_frq(train, test, cols=['rate'])
        new_feats.append(ff)

        new_feats.append(['rate'])

        tt_new_features = [x for item in new_feats for x in item]
        if test_zh_feats:
            new_feats.append(tt_new_features)
            return new_feats, zh_num_feats
        else:
            return tt_new_features, zh_num_feats

    def add_dk_features(clean_outlier=True, test_dk_feats=True):
        """
        deal with dk features!
        :param clean_outlier:
        :param test_dk_features:
        :return:
        """
        global train, test
        dk_num_features = ['DKFFE', 'DKYE']
        dk_ord_features = ['DKLL']
        new_feats = []

        ## 处理DK利率异常数据
        if clean_outlier:
            train[dk_num_features] = train[dk_num_features].apply(lambda x: x - 237)
            test[dk_num_features] = test[dk_num_features].apply(lambda x: x - 237)

            check_features = deal_dkll(train, test)
            new_feats.append(check_features)

        ## 业务特征
        ## 已还利息
        train['YHDK'] = train['DKFFE'] - train['DKYE']
        test['YHDK'] = test['DKFFE'] - test['DKYE']

        ff = encode_frq(train,test,cols=['YHDK'])
        new_feats.append(ff)
        new_feats.append(['YHDK'])

        tt_new_features = [x for item in new_feats for x in item]
        if test_dk_feats:
            new_feats.append(tt_new_features)
            return new_feats, dk_ord_features + dk_num_features
        else:
            return tt_new_features, dk_ord_features + dk_num_features

    new_person_features = add_person_features(dl_time=True)

    zh_features,base_zh_features = add_zh_features(test_zh_feats=False)
    dk_features,base_dk_features = add_dk_features(test_dk_feats=False)

    dw_features = add_dw_features(clean_outlier=False,test_dw_feats=False)
    
    model_params = load_params(seed)
    m_score = [[0.9474181440647862, 0.5618425825172289]]

    tt_features = new_person_features + dw_features + zh_features + base_zh_features + base_dk_features+dk_features

    mt = make_test(train, test, base_features=tt_features,
                                  new_features =[], m_score = m_score.copy(), label = label)

    mt.init_CV(seed,n_split=5)
    oof, _ = mt.lgb_test(lgb_params=model_params['lgb'])
    mt.submit()
