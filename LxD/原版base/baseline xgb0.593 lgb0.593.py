import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import lightgbm as lgb
import xgboost as xgb

from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from helper import tpr_weight_funtion, kfold_mean

warnings.filterwarnings('ignore')
# pd.set_option('max.columns', 100)


def load_data(data_path='../data/'):
    """加载数据"""
    data_train = pd.read_csv(data_path + 'train.csv')
    data_test = pd.read_csv(data_path + 'test.csv')
    submit = pd.read_csv(data_path + 'submit.csv')

    return data_train, data_test, submit


if __name__ == '__main__':
    """lgb、xgb同特征，xgb线上0.593，lgb线上0.593，可stacking"""

    data_train, data_test, submit = load_data()

    # ------------------------- feature engineering--------------------------
    data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)

    cate_2_cols = ['XINGBIE', 'ZHIWU', 'XUELI']
    cate_cols = ['HYZK', 'ZHIYE', 'ZHICHEN', 'DWJJLX', 'DWSSHY', 'GRZHZT']
    num_cols = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE']  # DKLL

    # -------------------------------- age ----------------------------------
    use_base = True
    if use_base:
        data['age'] = ((1609430399 - data['CSNY']) / (365 * 24 * 3600)).astype(int)
    else:
        long_time_mask = data['CSNY'].astype(str).str.len() == 12
        data['time'] = 0
        data.loc[long_time_mask, 'time'] = pd.to_datetime(data.loc[long_time_mask, 'CSNY'], unit='ms')
        data.loc[~long_time_mask, 'time'] = pd.to_datetime(data.loc[~long_time_mask, 'CSNY'], unit='s')
        data['time'] = pd.to_datetime(data.loc[:, 'time'].copy())
        data['age'] = 2020 - data['time'].dt.year
        data['time_tonow'] = (datetime.now() - data['time']).dt.days
        data.drop(['time', 'time_tonow'], axis=1, inplace=True)
    data.drop(['CSNY'], axis=1, inplace=True)

    # --------------------------------- 偏差修正 -------------------------------
    correct_error = False
    if correct_error:
        for col in ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE']:
            data[col] = data[col] - 237

    not_drop = []

    # --------------------------------- 业务特征 ---------------------------------

    data['JC_ratio'] = data['GRYJCE'] / data['GRJCJS']  # 缴存比例 = 个人月缴存额 / 个人缴存基数

    data['GRJCJS_lowest'] = data['GRYJCE'] / 0.12  # 根据月缴存额，计算正常的个人缴存基数范围
    data['GRJCJS_highest'] = data['GRYJCE'] / 0.05
    data['GRJCJS_1'] = data['GRJCJS_lowest'] - data['GRJCJS']
    data['GRJCJS_2'] = data['GRJCJS_highest'] - data['GRJCJS']

    # 当年缴存（猜测是4个月） - 当年归集
    data['DNTQ'] = (data['GRYJCE'] + data['DWYJCE']) * 4 - data['GRZHDNGJYE']

    # 【日照】贷款额度 = 申请人及配偶的个人月缴存额之和 / 实际缴存比例 * 12(月) * 0.45(还款能力系数) * 最长贷款年限
    data['until_retire'] = 65 - data['age']
    data['DK_years'] = data['until_retire'].apply(lambda x: x if x < 30 else 30)
    data['DKED_1'] = (data['GRYJCE'] + data['DWYJCE']) / data['JC_ratio'] * 12 * 0.45 * data['DK_years']
    data.drop(['until_retire', 'DK_years'], axis=1, inplace=True)

    data['DKFFE_DKYE'] = data['DKFFE'] - data['DKYE']  # 贷款发放额 - 贷款余额

    generate_fea = ['JC_ratio', 'GRJCJS_lowest', 'GRJCJS_highest', 'GRJCJS_1', 'GRJCJS_2', 'DNTQ', 'DKED_1',
                    'DKFFE_DKYE']

    # ----------------------------------- 类别 -----------------------------------

    # count encoding
    cat_col = ['HYZK', 'ZHIYE', 'ZHICHEN', 'ZHIWU', 'XUELI', 'DWJJLX', 'DWSSHY', 'GRZHZT']
    for col in cat_col:
        data[col + '_COUNT'] = data[col].map(data[col].value_counts())
        col_idx = data[col].value_counts()
        for idx in col_idx[col_idx < 10].index:
            data[col] = data[col].replace(idx, -1)

    # label encoding
    label_enc_cols = ['XINGBIE', 'HYZK', 'ZHIYE', 'ZHICHEN', 'ZHIWU', 'XUELI']
    for col in label_enc_cols:
        lbl = LabelEncoder()
        data[col] = lbl.fit_transform(data[col].astype(str))

    # target encoding
    target_enc_fea = ['DWJJLX', 'DWSSHY', 'GRZHZT']
    data = kfold_mean(data[~data['label'].isna()], data[data['label'].isna()], 'label', target_enc_fea)

    # 类别组合共现、类别偏好
    cate_cols_combine = [[cate_cols[i], cate_cols[j]] for i in range(len(cate_cols)) for j in
                         range(i + 1, len(cate_cols))]
    for f1, f2 in tqdm(cate_cols_combine):
        data['{}_{}_count'.format(f1, f2)] = data.groupby([f1, f2])['id'].transform('count')
        data['{}_in_{}_prop'.format(f1, f2)] = data['{}_{}_count'.format(f1, f2)] / data[f2 + '_COUNT']
        data['{}_in_{}_prop'.format(f2, f1)] = data['{}_{}_count'.format(f1, f2)] / data[f1 + '_COUNT']

    # ----------------------------- 训练集和测试集 -----------------------------
    train = data[data['label'].isna()==False].reset_index(drop=True)
    test = data[data['label'].isna()==True].reset_index(drop=True)
    print("train shape ", train.shape, "     test shape ", test.shape)

    drop_feats = [f for f in train.columns if
                  (train[f].nunique() <= 3) and (f not in (cate_2_cols + cate_cols + not_drop + ['label']))]
    print("nums of drop features ", len(drop_feats))

    print('Nan col nums of train: ', train.isnull().any().sum())
    print('Nan col nums of test: ', test.isnull().any().sum()-1)

    # ------------------------------- model ----------------------------------
    feature = [col for col in train.columns if col not in ['id', 'label'] + drop_feats]

    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    final_output = np.zeros(test.shape[0])
    score = []
    auc = []
    seeds = [1023, 2048, 2098]
    for seed in seeds:
        print('seed :', seed)
        num_folds = 5
        kfold = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True).split(
            train.drop(['label'], axis=1), train['label'])

        output_probs = np.zeros((test.shape[0], num_folds))  # 记录每折中对测试集的预测结果，最终取平均值作为最终预测
        valid_probs = np.zeros((train.shape[0], num_folds))
        for fold, (train_idx, valid_idx) in enumerate(kfold):
            X_train, y_train = train[feature].iloc[train_idx], train['label'].iloc[train_idx]
            X_valid, y_valid = train[feature].iloc[valid_idx], train['label'].iloc[valid_idx]

            # clf = lgb.LGBMClassifier(
            #     learning_rate=0.05,
            #     n_estimators=10230,
            #     num_leaves=31,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=1023,
            #     metric=None
            # )

            clf = xgb.XGBClassifier(
                learning_rate=0.05,
                n_estimators=10000,
                reg_alpha=0.5,
                reg_lambda=0.5,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                random_state=1023,
            )

            clf.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='auc',  # lambda y_true, y_pred: tpr_score(y_true, y_pred),
                    # categorical_feature = cate_2_cols + cate_cols,
                    early_stopping_rounds=200,
                    verbose=False)

            y_pred_valid = clf.predict_proba(X_valid)[:, 1]  # 验证集预测概率
            # y_pred_valid_label = [1 if p > 0.5 else 0 for p in y_pred_valid]    # 概率转类别（0、1）
            score.append(tpr_weight_funtion(y_valid, y_pred_valid))
            auc.append(roc_auc_score(y_valid, y_pred_valid))
            output_probs[:, fold] = clf.predict_proba(test[feature])[:, 1]  # 对测试集预测

        final_output = final_output + np.mean(output_probs, axis=1) / len(seeds)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d s' % (int((datetime.now() - start).seconds)))
    print('MEAN-AUC:%.6f, STD-AUC:%.6f' % (np.mean(auc), np.std(auc)))
    print('MEAN-Score:%.6f, STD-Score:%.6f' % (np.mean(score), np.std(score)))

    print(sum(final_output > 0.5))

    # submit['id'] = test['id']
    # submit['label'] = final_output
    # submit.to_csv('../result/' + 'xgb_1.csv', index=False)

