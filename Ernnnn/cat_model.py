## coding:utf-8
import datetime
import gc
import os
import random
import time
import warnings


import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from lianyhaii.feature_tools import encode_cat_feats, agg_num_feats



warnings.filterwarnings('ignore')


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()

    d['prob'] = list(y_predict)

    d['y'] = list(y_true)

    d = d.sort_values(['prob'], ascending=[0])

    y = d.y

    PosAll = pd.Series(y).value_counts()[1]

    NegAll = pd.Series(y).value_counts()[0]

    pCumsum = d['y'].cumsum()

    nCumsum = np.arange(len(y)) - pCumsum + 1

    pCumsumPer = pCumsum / PosAll

    nCumsumPer = nCumsum / NegAll

    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]

    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]

    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    print(f'0.1% tpr is {TR1}, 0.5% tpr is {TR2} ,1% tpr is {TR3}')
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3
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
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    if drop_zhiwu:
        # train = train[train['ZHIWU'] == 0]
        test = test[test['ZHIWU'] == 0]

    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    return train, test


def load_params(seed):
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',

        'n_jobs': -1,
        'learning_rate': 0.03,
        'num_leaves': 2 ** 8,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 10000,
        'max_bin': 255,
        'verbose': -1,
        'seed': seed,
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

    cat_params = {
        'n_estimators': 5000,
        'learning_rate': 0.07,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'random_seed': seed,
        # 'metric_period': 200,
        'od_wait': 500,
        # 'task_type': 'GPU',
        'depth': 8,
        'colsample_bylevel':0.8,
    }

    lr_params = {
        'random_state': seed,
        'C': 1,
        'max_iter': 1000,
        'n_jobs': -1,

    }

    model_params = {
        # 'lgb': lgb_params,
        # 'xgb': xgb_params,
        'cat':cat_params,
        # 'lr':lr_params,
    }

    return model_params


def test_feature(pre_features,new_features):
    print(new_features)
    mt = make_test(train, test, base_features=pre_features,
                   new_features=new_features, m_score=m_score.copy(), label=label)
    mt.init_CV(seed,shuffle=True)
    _, k = mt.cat_test(cat_params=model_params['cat'])


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
        self.features_imp = []

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
        f1_score_ = f1_score(y_true=self.train[self.label], y_pred=np.round(oof_predictions), average='macro')

        # acc_score = accuracy_score(y_true=self.train[self.label], y_pred=oof_predictions > 0.5)
        print('global auc :', auc_score)
        print('global tpr :', tpr_score)
        print('global f1  :', f1_score_)
        print('=' * 10 + 'different with previous version' + '=' * 10)
        print('diff of auc :', np.round(auc_score - self.m_score[-1][0], 5))
        print('diff of tpr :', np.round(tpr_score - self.m_score[-1][1], 5))
        print('diff of f1 :', np.round(f1_score_ - self.m_score[-1][2], 5))
        self.m_score.append([auc_score, tpr_score, f1_score_])


    def cat_test(self, cat_params, cv_score=False):
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))
        cat_features = [x for x in self.train.select_dtypes(include=['category']).columns.tolist() if
                        x in self.features]

        for n, (trn, val) in enumerate(self.CV_type.split(self.train, self.train[self.label])):
            trn_X, trn_y = self.train.loc[trn, self.features], self.train.loc[trn, self.label]
            val_X, val_y = self.train.loc[val, self.features], self.train.loc[val, self.label]

            estimator = CatBoostClassifier(**cat_params)
            estimator.fit(
                trn_X, trn_y,
                cat_features=cat_features,
                # early_stopping_rounds=100,
                eval_set=[(trn_X, trn_y), (val_X, val_y)],
                use_best_model=True,
                metric_period=500,
                verbose=True,
            )

            oof_predictions[val] = estimator.predict_proba(val_X)[:, 1]

            if cv_score:
                cv_score_list.append(roc_auc_score(y_true=trn_y, y_score=estimator.predict_proba(trn_X)[:, 1]))
            tt_predicts += estimator.predict_proba(self.test[self.features])[:, 1] / self.CV_type.n_splits

        self.__check_diff_score(oof_predictions)

        self.predictions = tt_predicts
        if cv_score:
            return oof_predictions, tt_predicts, cv_score_list
        else:
            return oof_predictions, tt_predicts


    def submit(self, ID, sub_file=True, threshold=None):
        today = time.strftime("%Y-%m-%d", time.localtime())[5:]
        # self.test[self.label] = [int(x) for x in self.predictions > 0.5]
        if threshold is None:
            self.test[self.label] = self.predictions
        else:
            self.test[self.label] = [int(x > threshold) for x in self.predictions]

        sub_train = self.test[[ID, self.label]].copy()
        if sub_file:
            sub_test = pd.read_csv('./data/submit.csv')
            sub = sub_test[[ID]].merge(sub_train, on=ID, how='left')
        else:
            sub = sub_train.copy()

        print('null in sub', '\n', sub.isnull().sum())
        sub.fillna(0, inplace=True)
        score = str(np.round(self.m_score[-1][0], 4)) + "_" + str(np.round(self.m_score[-1][1], 4))
        sub.to_csv(f'sub_{today}_{score}.csv', index=False)

def category_encoding(train, test, cols):
    category_feats = []
    for col in cols:
        col_cat = pd.Categorical(train[col].append(test[col]))
        f_name = f"{col}_cate_enc"
        train[f_name] = train[col].astype(col_cat)
        test[f_name] = test[col].astype(col_cat)
        category_feats.append(f_name)
    return category_feats
def label_encoding(train, test, cols,inplace=False,keep_na=False):

    lb_feats = []
    for col in cols:
        if not inplace:
            f_name = f'{col}_lb_enc'
        else:
            f_name = col

        le = LabelEncoder()
        le.fit(train[col].append(test[col]))

        train[f_name] = le.transform(train[col])
        test[f_name] = le.transform(test[col])

        lb_feats.append(f_name)

    return lb_feats

def freq_encode(df1, df2, cols):
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


def agg_norm_feats(df1, df2, cat_feats, num_feats):
    norm_feats = []
    for f1 in cat_feats:
        tmp_df = pd.concat([df1[[f1] + num_feats], df2[[f1] + num_feats]], axis=0, ignore_index=True)
        gp = tmp_df.groupby(f1)
        for f2 in num_feats:
            tmp_df_mean_dict = gp[f2].mean()
            tmp_df_std_dict = gp[f2].std()

            f_name = f'{f2}_norm_by_{f1}'
            df1[f_name] = (df1[f2] - df1[f1].map(tmp_df_mean_dict)) / (
                df1[f1].map(tmp_df_std_dict))

            df2[f_name] = (df2[f2] - df2[f1].map(tmp_df_mean_dict)) / (
                df2[f1].map(tmp_df_std_dict))

            norm_feats.append(f_name)
    return norm_feats

## 自定义eval
if __name__ == '__main__':

    seed = 0
    seed_everything(seed)
    drop_zhiwu = True

    label = 'label'
    ID = 'id'

    train, test = load_data()
    feature_select = False

    remove_feat = ['id', label, 'GRYJCE', 'HYZK', 'ZHIWU', 'XUELI', 'ZHIYE', 'ZHICHEN',
                   'noise_sample', 'time', 'CSNY', 'month']

    def add_features_base(test_feats=True):
        global train,test
        new_feats = []

        tt_new_features = [x for item in new_feats for x in item]
        if test_feats:
            new_feats.append(tt_new_features)
            return new_feats
        else:
            return tt_new_features
    def add_person_features(dl_time=True):
        global train, test
        # person_cat_feats = ['XINGBIE', 'ZHIYE', 'ZHICHEN']
        # bin_cat_feats = ['XINGBIE', 'ZHICHEN']
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
    def add_dw_features(clean_outlier=False,test_dw_feats=True):
        global train, test

        if clean_outlier:
            train['DWSSHY'] = train['DWSSHY'].map(lambda x:np.nan if x in [19,20] else x)
            test['DWSSHY'] = test['DWSSHY'].map(lambda x:np.nan if x in [19,20] else x)

        new_feats = []

        ctg_feats = label_encoding(train,test,cols=['DWJJLX','DWSSHY'],)
        new_feats.append(ctg_feats)

        ctg_feats = freq_encode(train,test,cols=['DWJJLX','DWSSHY'])
        new_feats.append(ctg_feats)

        tt_new_features = [x for item in new_feats for x in item]
        if test_dw_feats:
            new_feats.append(tt_new_features)
            return new_feats
        else:
            return tt_new_features
    def add_zh_features(clean_outlier=True, test_zh_feats=True):
        """
        how to deal with zh features?
        :param encode_type:
        :return: features list
        """
        global train, test
        zh_num_feats = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'DWYJCE']
        zh_cat_feats = ['GRZHZT']
        new_feats = []

        ## clean outlier
        if clean_outlier:
            train[zh_num_feats] = train[zh_num_feats].apply(lambda x: x - 237)
            test[zh_num_feats] = test[zh_num_feats].apply(lambda x: x - 237)

        ## deal with cat encode
        ctg_feats = category_encoding(train,test,zh_features)
        new_feats.append(ctg_feats)

        ##业务特征
        ye_features = []
        train['DWJCBL'] = train['DWYJCE'] / (train['GRJCJS'])
        test['DWJCBL'] = test['DWYJCE'] / (test['GRJCJS'])

        ye_features.append('DWJCBL')

        train['rate'] = np.round(train['GRZHDNGJYE'] / (train['DWYJCE'] * 2), 4)
        test['rate'] = np.round(test['GRZHDNGJYE'] / (test['DWYJCE'] * 2), 4)
        ye_features.append('rate')
        new_feats.append(ye_features)
        # restart features
        # new_feats = [[x for item in new_feats for x in item]]

        # ##对数值特征进行cat化
        freq_feats = freq_encode(train,test,zh_num_feats+ye_features)
        new_feats.append(freq_feats)

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

            #corrected DKFFE
            # train.loc[train['DKFFE']>300000,'DKFFE'] -= 10000
            # test.loc[test['DKFFE']>300000,'DKFFE'] -= 10000

        ## 业务特征
        ## 已还利息
        train['YHDK'] = train['DKFFE'] - train['DKYE']
        test['YHDK'] = test['DKFFE'] - test['DKYE']
        new_feats.append(['YHDK'])
        frq_features = freq_encode(train,test,['YHDK'])
        new_feats.append(frq_features)

        tt_new_features = [x for item in new_feats for x in item]
        if test_dk_feats:
            new_feats.append(tt_new_features)
            return new_feats, dk_ord_features + dk_num_features
        else:
            return tt_new_features, dk_ord_features + dk_num_features
    p_features = add_person_features(dl_time=True)
    dw_features = add_dw_features(test_dw_feats=False)
    zh_features,base_zh_features = add_zh_features(test_zh_feats=False)
    dk_features,base_dk_features = add_dk_features(test_dk_feats=False)

    def add_zh_dk_features(test_zh_dk_features=True):
        global train,test
        new_feats = []

        ##load loan
        loan_df = pd.read_csv('./data/data.csv')
        loan_df['DKLL'] = np.round(loan_df['DKLL'],3)

        train = train.merge(loan_df,on=['DKFFE','DKYE','DKLL'],how='left')
        test = test.merge(loan_df,on=['DKFFE','DKYE','DKLL'],how='left')

        ##逐步加入变量
        check_features = []
        for t in [0.1,0.01]:
            train[f'DKWC_check_{str(t)}'] = (train['DKWC'] < t).astype(int)
            test[f'DKWC_check_{str(t)}'] = (test['DKWC'] < t).astype(int)
            check_features.append(f"DKWC_check_{str(t)}")

        new_feats.append(check_features)
        ff = agg_norm_feats(train,test,cat_feats=['DWJJLX'],num_feats=['DWJCBL','DKFFE','GRZHYE','DWYJCE'])
        new_feats.append(ff)

        ff = agg_norm_feats(train,test,cat_feats=['DWSSHY'],num_feats=['DWJCBL','DKFFE','GRZHYE','DWYJCE'])
        new_feats.append(ff)


        tt_new_features = [x for item in new_feats for x in item]
        if test_zh_dk_features:
            new_feats.append(tt_new_features)
            return new_feats
        else:
            return tt_new_features


    group_features = add_zh_dk_features(test_zh_dk_features=False)


    model_params = load_params(seed)
    m_score = [[0.9499966161516394, 0.5777656873413131,0]]

    # baseline 0.584528
    tt_features = p_features + dw_features + base_zh_features + zh_features + base_dk_features + dk_features + \
                  group_features


    # for group in group_features:
    #     test_feature(tt_features, group)
    mt = make_test(train, test, base_features=tt_features,
                                  new_features = [], m_score = m_score.copy(), label = label)

    mt.init_CV(seed,n_split=5)
    oof, k = mt.cat_test(cat_params=model_params['cat'])
    # mt.submit(ID,)
    # pd.DataFrame({
    #     ID:train[ID],
    #     label:train[label],
    #     'predict':oof
    # }).to_csv('./ensemble/cat_best_oof.csv',index=False)

