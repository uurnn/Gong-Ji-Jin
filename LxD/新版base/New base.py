import numpy as np
import pandas as pd
import warnings
import math

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from model_for_predict import Model
from woe import Woe
from scipy.special import boxcox1p
from k_means import K_means
from scipy.stats import entropy
from helper import kfold_risk_feature

warnings.filterwarnings('ignore')
pd.set_option('max.columns', 30)


def load_data(data_path='../data/'):
    """加载数据"""

    data_train = pd.read_csv(data_path + 'train.csv')
    data_test = pd.read_csv(data_path + 'test.csv')
    submit = pd.read_csv(data_path + 'submit.csv')

    return data_train, data_test, submit


def correct_num_fea(trn, tst, correct_bias=True, correct_dkll=False):

    if correct_bias:
        feats = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE']
        for df in [trn, tst]:
            for fea in feats:
                df[fea] = df[fea] - 237

    if correct_dkll:
        for df in [trn, tst]:
            df['DKLL'] = np.round(df['DKLL'], 3)
            year5_M_rate_mask = df['DKLL'].isin([3.25, 3.025])
            year1_M_rate_mask = df['DKLL'].isin([2.750])
            year5_M2_rate_mask = df['DKLL'].isin([3.575])
            df.loc[year5_M_rate_mask, 'DKLL'] = 2.708
            df.loc[year1_M_rate_mask, 'DKLL'] = 2.292
            df.loc[year5_M2_rate_mask, 'DKLL'] = 2.979

            # double_house_mask = df['DKLL'].isin([2.979, 2.521])
            # df['DKLL_check'] = (year5_M_rate_mask | year5_M2_rate_mask | year1_M_rate_mask).astype(int)
            # df['double_house'] = (double_house_mask).astype(int)

    return trn, tst


def compute_age(trn, tst, use_base=True):

    for df in [trn, tst]:
        if use_base:
            df['age'] = ((1609430399 - df['CSNY']) / (365 * 24 * 3600)).astype(int)
        else:
            long_time_mask = df['CSNY'].astype(str).str.len() == 12
            df['time'] = 0
            df.loc[long_time_mask, 'time'] = pd.to_datetime(df.loc[long_time_mask, 'CSNY'], unit='ms')
            df.loc[~long_time_mask, 'time'] = pd.to_datetime(df.loc[~long_time_mask, 'CSNY'], unit='s')
            df['time'] = pd.to_datetime(df.loc[:, 'time'].copy())
            df['age'] = 2020 - df['time'].dt.year
            df['time_tonow'] = (datetime.now() - df['time']).dt.days
            df.drop(['time', 'time_tonow'], axis=1, inplace=True)

        df.drop(['CSNY'], axis=1, inplace=True)

    return trn, tst


def bin_feature(trn, tst, ):
    """特征分桶"""

    for df in [trn, tst]:
        # 原始数值特征
        df['GRJCJS'] = pd.cut(df['GRJCJS'],
                              [-np.inf, 2000, 4000, 6000, 8000, 10000, 12000, np.inf], labels=False)
        df['GRZHYE'] = pd.cut(df['GRZHYE'],
                              [-np.inf, 10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, np.inf], labels=False)
        df['GRZHSNJZYE'] = pd.cut(df['GRZHSNJZYE'],
                                  [-np.inf, 10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, np.inf], labels=False)
        df['GRZHDNGJYE'] = pd.cut(df['GRZHDNGJYE'],
                                  [-np.inf, -30000, -20000, -10000, -5000, 0, 5000, 10000, np.inf], labels=False)
        df['GRYJCE'] = pd.cut(df['GRYJCE'],
                              [-np.inf, 200, 400, 600, 800, 1000, 1200, 1400, np.inf], labels=False)
        df['DWYJCE'] = pd.cut(df['DWYJCE'],
                              [-np.inf, 200, 400, 600, 800, 1000, 1200, 1400, np.inf], labels=False)
        df['DKFFE'] = pd.cut(df['DKFFE'],
                             [-np.inf, 50000, 100000, 150000, 200000, 250000, 300000, np.inf], labels=False)
        df['DKYE'] = pd.cut(df['DKYE'],
                            [-np.inf, 50000, 100000, 150000, 200000, 250000, 300000, np.inf], labels=False)

    return trn, tst


def static_rows(trn, tst):
    """按行统计，如 0 值、缺失值、均值、方差等"""

    cols = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE']

    trn['nums_zero'] = (trn[cols] == 0).astype(int).sum(axis=1)
    tst['nums_zero'] = (tst[cols] == 0).astype(int).sum(axis=1)

    return trn, tst


def static_feature(trn, tst, whole=False):
    """统计量，xgb上升"""

    if not whole:       # 按train统计，map到test，xgb线下、线上均升，    --------- 可试train和test分别独立统计，而不是map

        cat_col = ['DWSSHY', 'DWJJLX']
        gen_col = ['DNTQ', 'GRJCJS_1', 'GRJCJS_2', 'DKED_1', 'DKFFE_DKYE']       # mean 线上下降
        num_col = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DKFFE', 'DKYE']
        for col_1 in cat_col:
            for col_2 in num_col:
                mean_dic = trn.groupby([col_1])[col_2].mean().reset_index().set_index(col_1)[col_2].to_dict()
                trn[col_1 + '_' + col_2 + '_mean'] = trn[col_1].apply(lambda x: mean_dic[x])
                tst[col_1 + '_' + col_2 + '_mean'] = tst[col_1].apply(lambda x: mean_dic[x])

                # max_min_dic = trn.groupby([col_1])[col_2].agg(lambda x: max(x) - min(x)).reset_index().set_index(col_1)[col_2].to_dict()
                # trn[col_1 + '_' + col_2 + '_max_min'] = trn[col_1].apply(lambda x: max_min_dic[x])
                # tst[col_1 + '_' + col_2 + '_max_min'] = tst[col_1].apply(lambda x: max_min_dic[x])

    else:   # 全局统计，下降
        data = pd.concat([trn, tst], axis=0).reset_index(drop=True)
        cat_col = ['DWSSHY', 'DWJJLX']
        num_col = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DKFFE', 'DKYE']
        for col_1 in cat_col:
            for col_2 in num_col:
                data[col_1 + '_' + col_2 + '_mean'] = data.groupby(col_1)[col_2].transform('mean')

        # get train & test data
        trn = data[data['label'].isna() == False].reset_index(drop=True)
        tst = data[data['label'].isna() == True].reset_index(drop=True)

    return trn, tst


def jc_fea(trn, tst):
    """缴存特征"""

    for df in [trn, tst]:
        df['JC_ratio'] = df['GRYJCE'] / df['GRJCJS']
        df['DNTQ'] = (df['GRYJCE'] + df['DWYJCE']) * 4 - df['GRZHDNGJYE']

        # ------------------------------ new ↓↓ --------------------------------

        # # xgb上升, lgb波动略降。可单独试
        # df['DNGJ_time'] = np.round(df['GRZHDNGJYE'] / (df['DWYJCE'] + df['GRYJCE']), 4)

        # df['GRJCJS_age'] = df['GRJCJS'] / df['age']       # xgb略升，拨动极小

        # df['GRZHYE_is_0'] = df['GRZHYE'].apply(lambda x: 1 if x == 0 else 0)
        # df['SNJZ_is_0'] = df['GRZHSNJZYE'].apply(lambda x: 1 if x == 0 else 0)
        # df['DKYE_is_0'] = df['DKYE'].apply(lambda x: 1 if x == 0 else 0)

    generate_fea = ['JC_ratio', 'DNTQ']

    return trn, tst, generate_fea


def dk_fea(trn, tst):
    """贷款特征"""

    for df in [trn, tst]:

        df['GRJCJS_1'] = df['GRYJCE'] / 0.12 - df['GRJCJS']
        df['GRJCJS_2'] = df['GRYJCE'] / 0.05 - df['GRJCJS']

        # 【日照】贷款额度 = 申请人及配偶的个人月缴存额之和 / 实际缴存比例 * 12(月) * 0.45(还款能力系数) * 最长贷款年限
        df['until_retire'] = 65 - df['age']
        df['DK_years'] = df['until_retire'].apply(lambda x: x if x < 30 else 30)
        df['DKED_1'] = (df['GRYJCE'] + df['DWYJCE']) / df['JC_ratio'] * 12 * 0.45 * df['DK_years']
        df.drop(['until_retire', 'DK_years'], axis=1, inplace=True)

        df['DKFFE_DKYE'] = df['DKFFE'] - df['DKYE']       # 线下lgb下降，xgb上升！！！

        # ------------------------------ new ↓↓ --------------------------------

    generate_fea = ['GRJCJS_1', 'GRJCJS_2', 'DKED_1', 'DKFFE_DKYE']

    return trn, tst, generate_fea


def deal_category(trn, tst):
    """处理类别特征"""

    # drop feature --- different between train and test
    drop_cate = ['XUELI', 'HYZK', 'ZHIWU']
    for df in [trn, tst]:
        df.drop(drop_cate, axis=1, inplace=True)

    # whole data
    data = pd.concat([trn, tst], axis=0).reset_index(drop=True)

    data['GRZHZT'] = data['GRZHZT'].apply(lambda x: 2 if x in [2, 4, 5] else x)

    # count encoding
    cat_col = ['ZHIYE', 'ZHICHEN', 'DWJJLX', 'DWSSHY', 'GRZHZT']
    for col in cat_col:
        data[col + '_COUNT'] = data[col].map(data[col].value_counts())
        col_idx = data[col].value_counts()
        for idx in col_idx[col_idx < 10].index:
            data[col] = data[col].replace(idx, -1)

    # label encoding
    label_enc_cols = ['XINGBIE', 'ZHIYE', 'ZHICHEN']
    for col in label_enc_cols:
        lbl = LabelEncoder()
        data[col] = lbl.fit_transform(data[col].astype(str))

    # # target encoding   --- 线下auc下降，tpr上升
    # target_enc_fea = ['DWJJLX', 'DWSSHY', 'GRZHZT']
    # data = kfold_mean(data[~data['label'].isna()], data[data['label'].isna()], 'label', target_enc_fea)

    # 类别组合共现、类别偏好
    cate_cols_combine = [[cat_col[i], cat_col[j]] for i in range(len(cat_col)) \
                         for j in range(i + 1, len(cat_col))]
    for f1, f2 in cate_cols_combine:
        data['{}_{}_count'.format(f1, f2)] = data.groupby([f1, f2])['id'].transform('count')
        data['{}_in_{}_prop'.format(f1, f2)] = data['{}_{}_count'.format(f1, f2)] / data[f2 + '_COUNT']
        data['{}_in_{}_prop'.format(f2, f1)] = data['{}_{}_count'.format(f1, f2)] / data[f1 + '_COUNT']

    # get train & test data
    trn = data[data['label'].isna()==False].reset_index(drop=True)
    tst = data[data['label'].isna()==True].reset_index(drop=True)

    # # k fold target encoding
    # target_encode_cols = ['DWJJLX', 'DWSSHY']
    # trn, tst = kfold_risk_feature(trn, tst, target_encode_cols, 5, 1023)

    return trn, tst.drop('label', axis=1)


def rank_feature(trn, tst, rank_cols):

    data = pd.concat([trn, tst], axis=0).reset_index(drop=True)

    # ------------------------------ 数值排序特征 --------------------------------

    for col in rank_cols:
        dic = {}
        for i, v in enumerate(sorted(list(set(data[col].values)))):
            dic[v] = i + 1
        data[col] = data[col].map(dic)

    # get train & test data
    trn = data[data['label'].isna() == False].reset_index(drop=True)
    tst = data[data['label'].isna() == True].reset_index(drop=True)

    return trn, tst.drop('label', axis=1)


def woe_encoding(trn, tst):

    auto_col_bins = {'JC_ratio': 9}        # 定义woe编码目标及分桶数量
    cat_woe_cols = []

    woe = Woe(trn, tst, auto_col_bins=auto_col_bins, cat_woe_cols=cat_woe_cols)

    # woe.plot_iv(['GRJCJS'])

    trn, tst = woe.get_woe()

    return trn, tst


def encode_frq(df1, df2, cols):
    new_cols = []
    for col in cols:
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=False, normalize=False).to_dict()
        vc[-1] = -1
        nm = col + '_FrqEnc'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        # print(nm, ', ', end='\n')
        new_cols.append(nm)
    return df1, df2, new_cols


if __name__ == '__main__':

    trn, tst, sub = load_data()

    # ---- feature engineering -----

    cate_2_cols = ['XINGBIE', 'ZHIWU', 'XUELI']
    cate_cols = ['HYZK', 'ZHIYE', 'ZHICHEN', 'DWJJLX', 'DWSSHY', 'GRZHZT']
    num_cols = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE']  # DKLL

    # 数据修正
    trn, tst = correct_num_fea(trn, tst, correct_bias=True, correct_dkll=False)      # True False

    # age
    trn, tst = compute_age(trn, tst, use_base=True)     # True

    # 缴存特征
    trn, tst, gen_fea_1 = jc_fea(trn, tst)

    # 贷款特征
    trn, tst, gen_fea_2 = dk_fea(trn, tst)

    num_cols = num_cols + gen_fea_1 + gen_fea_2

    # category
    trn, tst = deal_category(trn, tst)

    # # 行统计量
    # trn, tst = static_rows(trn, tst)      # 线下略降

    # ----------

    # 借鉴梁base
    trn, tst, freq_cols_1 = encode_frq(trn, tst, cols=['JC_ratio', 'GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'DWYJCE'])

    # new
    trn, tst, freq_cols_2 = encode_frq(trn, tst, cols=['DNTQ', 'GRJCJS_1', 'GRJCJS_2', 'DKED_1', 'DKFFE_DKYE'])

    # ----------

    # static train, test, --> map
    trn, tst = static_feature(trn, tst, whole=False)

    # 原始特征分桶
    trn, tst = bin_feature(trn, tst)

    # 衍生特征rank
    rank_cols = gen_fea_1 + gen_fea_2       # # + freq_cols_1 + freq_cols_2
    trn, tst = rank_feature(trn, tst, rank_cols)

    # # 数值特征分桶后 target encoding，xgb线下略升，波动不大，没试
    # target_encode_cols = num_cols
    # trn, tst = kfold_risk_feature(trn, tst, target_encode_cols, 5, 1023)

    # ----- model -----

    print('Nan col nums of train: ', trn.isnull().any().sum())
    print('Nan col nums of test:  ', tst.isnull().any().sum())

    feature = [col for col in trn.columns if col not in ['id', 'label']]

    print('Use features num: ', len(feature))
    print('featues: ', feature)

    model = Model(trn, tst, sub, feature, 'lgb')
    y_pred = model._predict()
    # model._submit('xgb_new_add_111111111111111.csv')

    """
    lgb:  原始特征分桶  衍生特征rank 统计特征 梁base-freq 线上未测试
    MEAN-AUC:0.952179, STD-AUC:0.005190
    MEAN-Score:0.565001, STD-Score:0.023522
    
    xgb:  原始特征分桶 衍生特征rank 统计特征 梁base-freq 线上0.593585   1180逾期
    MEAN-AUC:0.952901, STD-AUC:0.005014
    MEAN-Score:0.571240, STD-Score:0.022085
    """


