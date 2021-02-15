import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def kfold_risk_feature(train, test, feats, k, seed):
    """
    k fold target encoding，高基数类无序特征

    eg:
    target_encode_cols = ['col_1', 'col_2']
    train, test = kfold_risk_feature(train, test, target_encode_cols, k, seed)
    """

    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)       # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test


def kfold_mean(df_train, df_test, target, target_mean_list):
    """k-fold target encoding"""
    folds = StratifiedKFold(n_splits=5)
    mean_of_target = df_train[target].mean()
    for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(df_train, y=df_train['label']))):
        tr_x = df_train.iloc[trn_idx, :]
        vl_x = df_train.iloc[val_idx, :]
        for col in target_mean_list:
            df_train.loc[vl_x.index, f'{col}_target_enc'] = vl_x[col].map(
                tr_x.groupby(col)[target].mean())
    for col in target_mean_list:
        df_train[f'{col}_target_enc'].fillna(mean_of_target, inplace=True)

        df_test[f'{col}_target_enc'] = df_test[col].map(
            df_train.groupby(col)[f'{col}_target_enc'].mean())

        df_test[f'{col}_target_enc'].fillna(mean_of_target, inplace=True)
    return pd.concat([df_train, df_test], ignore_index=True)


def tpr_weight_funtion(y_true, y_predict):
    """线上评分函数"""
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
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]

    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3