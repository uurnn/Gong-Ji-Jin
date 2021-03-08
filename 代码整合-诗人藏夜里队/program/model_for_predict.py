import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from catboost import CatBoostClassifier
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from helper import tpr_weight_funtion


class Model(object):

    def __init__(self, train, test, submit, feature, model, model_seed=1023):
        self.train = train
        self.test = test
        self.submit = submit
        self.feature = feature          # 选择使用的feature

        self.finial_output = np.zeros(test.shape[0])
        self.finial_output_train = np.zeros(train.shape[0])

        self.kfold_train = np.zeros(train.shape[0])           # 新增

        self.auc = []
        self.score = []

        self.feature_importance_df = pd.DataFrame()

        try:
            assert model in ['lgb', 'xgb', 'rf']
        except AssertionError:
            model = 'lgb'

        if model == 'lgb':
            self.clf = lgb.LGBMClassifier(
                learning_rate=0.05,
                n_estimators=10230,
                num_leaves=64,            # 原 31
                subsample=0.8,            #
                colsample_bytree=0.75,    # 原0.8
                random_state=model_seed,
                metric=None,

                # min_child_samples=20,    # 默认
                # min_child_weight=0.001,  # 默认

                # reg_alpha=0.01,
                # reg_lambda=0.01,

                # # 网格搜索调参后，效果下降
                # max_depth=12,
                # num_leaves=40,
                # min_child_samples=20,
                # min_child_weight=0.001,
                # colsample_bytree=0.7,
                # subsample=0.8,
                # reg_alpha=0.08,
                # reg_lambda=7
            )
        elif model == 'xgb':
            self.clf = xgb.XGBClassifier(
                learning_rate=0.05,
                n_estimators=10000,
                reg_alpha=0.5,
                reg_lambda=0.5,
                subsample=0.8,
                colsample_bytree=0.8,          # 0.7     原0.8
                colsample_bylevel=0.8,         # 0.7     原0.8
                random_state=model_seed,

                # # 调参后，效果下降
                # max_depth=5,             # 7
                # min_child_weight=1,
                # gamma=0.3,
                # subsample=0.8,
                # colsample_bytree=0.7,
                # reg_alpha=0.7,
                # reg_lambda=0.5,
            )

    def _predict(self, seeds=(1023, 2048, 2098)):

        start = datetime.now()
        print(start.strftime('%Y-%m-%d %H:%M:%S'))
        for seed in seeds:
            print('seed : ', seed)
            num_folds = 5
            kfold = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True).split(
                        self.train.drop(['label'], axis=1), self.train['label'])

            output_probs = np.zeros((self.test.shape[0], num_folds))  # 记录每折中对测试集的预测结果，最终取平均值作为最终预测
            valid_probs = np.zeros((self.train.shape[0], num_folds))
            temp = np.zeros(self.train.shape[0])
            for fold, (train_idx, valid_idx) in enumerate(kfold):
                X_train, y_train = self.train[self.feature].iloc[train_idx], self.train['label'].iloc[train_idx]
                X_valid, y_valid = self.train[self.feature].iloc[valid_idx], self.train['label'].iloc[valid_idx]

                self.clf.fit(X_train, y_train,
                            eval_set=[(X_valid, y_valid)],
                            eval_metric='auc',
                            early_stopping_rounds=200,
                            verbose=False)

                y_pred_valid = self.clf.predict_proba(X_valid)[:, 1]  # 验证集预测概率
                temp[valid_idx] = y_pred_valid
                # y_pred_valid_label = [1 if p > 0.5 else 0 for p in y_pred_valid]    # 概率转类别（0、1）
                self.score.append(tpr_weight_funtion(y_valid, y_pred_valid))
                self.auc.append(roc_auc_score(y_valid, y_pred_valid))
                output_probs[:, fold] = self.clf.predict_proba(self.test[self.feature])[:, 1]  # 对测试集预测
                valid_probs[:, fold] = self.clf.predict_proba(self.train[self.feature])[:, 1]

                # feature importance
                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = list(filter(lambda x: x not in ['label', 'id'], X_train.columns))
                fold_importance_df["importance"] = self.clf.feature_importances_
                fold_importance_df["fold"] = fold + 1
                self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)

            self.finial_output = self.finial_output + np.mean(output_probs, axis=1) / len(seeds)
            self.finial_output_train = self.finial_output_train + np.mean(valid_probs, axis=1) / len(seeds)
            self.kfold_train = self.kfold_train + temp / len(seeds)

        # self.train['xgb_pred'] = self.kfold_train
        # self.train[['id', 'xgb_pred']].to_csv('../result/xgb_train_result.csv', index=False)

        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('time costed is: %d s' % (int((datetime.now() - start).seconds)))
        print('MEAN-AUC:%.6f, STD-AUC:%.6f' % (np.mean(self.auc), np.std(self.auc)))
        print('MEAN-Score:%.6f, STD-Score:%.6f' % (np.mean(self.score), np.std(self.score)))
        print("Nums of label > 0.5: ", sum(self.finial_output > 0.5))

        return self.finial_output, self.finial_output_train

    def importance(self, nums=50):
        """返回importance df 和 前nums重要的特征"""

        feature_importance_df_ = self.feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False)
        cols = list(self.feature_importance_df.iloc[:nums]['feature'].values)

        return feature_importance_df_, cols

    def _submit(self, name='base.csv'):
        self.submit['id'] = self.test['id']
        self.submit['label'] = self.finial_output
        self.submit.to_csv('../result/' + name, index=False)
