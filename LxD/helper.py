import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


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