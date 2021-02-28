# coding:utf-8
import time

import pandas as pd
import numpy as np

import pickle

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lianyhaii.model import tpr_weight_funtion

ID = 'id'
label = 'label'

lgb_598 = np.load('./user_data/lgb598_model.npy',allow_pickle=True)


def get_data_model(file:str):
    print(f'load data and models : {file}')

    path = './user_data/'
    train = pd.read_csv(path+f'{file}_train.csv')
    test = pd.read_csv(path+f'{file}_test.csv')
    models = np.load(path+f'{file}_model.npy',allow_pickle=True)
    for col in train.columns:
        if '_cate' in col:
            # c_ = pd.Categorical(train[col].append(test[col]))
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
            # train[col] = train[col].astype(c_)
            # test[col] = test[col].astype(c_)


    return train,test,models

train,test,models = get_data_model('lgb598')
train1,test1,models1 = get_data_model('cat599')
def get_model_name(model):
    return str(model).split('.')[0][1:]
print(get_model_name(models1[0]))
print(get_model_name(models[0]))

class pseudo_model():

    def __init__(self,train:pd.DataFrame,test:pd.DataFrame,models:np.array,label:str):
        print('data size ',train.shape,test.shape)
        self.train = train
        self.test = test
        self.models = models
        self.label = label
        self.features = [x for x in test.columns if x not in ['id',label]]



    def get_model_predict(self,nfold=5):

        tt_preds = np.zeros(self.test.shape[0])

        for model in self.models:
            name = get_model_name(model)
            print(f'getting prediction {name} ...')
            if name == 'lightgbm':

                tt_preds += model.predict(self.test[self.features],) / nfold
            elif name == 'catboost':
                tt_preds += model.predict_proba(self.test[self.features])[:,1] / nfold
        return tt_preds

    def fliter_sample(self,tt_preds:pd.Series,method='quanlity'):
        tt = self.test.copy()
        tt['label_score'] = tt_preds
        if method == 'quanlity':
            tt.sort_values('label_score',ascending=True,inplace=True)
            ## 取百分之1的数据进行打标签
            threshold = int(0.02 * self.test.shape[0])

            pre = tt.head(threshold*5).copy()
            pre[self.label+'_pseudo'] = 0
            last = tt.tail(threshold).copy()
            last[self.label+'_pseudo'] = 1


        elif method =='absolute':
            pre = tt.loc[tt['label_score']<0.0005,:].copy()
            pre[self.label+'_pseudo'] = 0
            last = tt.loc[tt['label_score']>=0.90,:].copy()
            last[self.label+'_pseudo'] = 1
            # new_sample = pd.concat([pre,last])

        else:
            raise ValueError('no support for this method')

        res = pre[['label_score',self.label+'_pseudo']].append(last[['label_score',self.label+'_pseudo']])

        return res

    def get_common_sample(self,new_samples,method='voting'):

        tt_new_samples = pd.DataFrame(index=self.test.index)
        common_sample_dict = {}
        for n,sample in enumerate(new_samples):
                tt_new_samples[f'score_{n}'] = sample['label_score'].copy()

        if method == 'voting':
            ## 至少有两个投票才计算入内
            tt_new_samples['score'] = tt_new_samples.iloc[:,tt_new_samples.columns.str.contains('score_')].notnull().sum(1)

            tt_new_samples = tt_new_samples[(tt_new_samples['score']>=2)]
            common_sample_dict = tt_new_samples['score'].map(lambda x:int(x)).to_dict()
        sample = new_samples[0]
        sample['score'] = sample.index.map(common_sample_dict)
        sample = sample[sample['score'].notnull()]
        common_sample_dict = sample[self.label+'_pseudo'].to_dict()

        return common_sample_dict

    def refit_model(self,common_sample_dict,seed):
        ## map label to test dataset
        tt = self.test.copy()
        tt['label'] = tt.index.map(common_sample_dict)
        tt = tt[~tt['label'].isnull()]
        print(f'add dataset size :{tt.shape}')
        ## merge dataset
        new_train = pd.concat([self.train,tt],ignore_index=True)
        # cat_features = [x for x in new_train.select_dtypes(include=['category']).columns.tolist() if
        #                 x in self.features]
        cat_features = [x for x in new_train.columns if '_cate' in x]
        for col in cat_features:
            new_train[col] = new_train[col].astype('category')
            self.test[col] = self.test[col].astype('category')

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        tt_preds = np.zeros(self.test.shape[0])
        oof_preds = np.zeros(new_train.shape[0])
        for n,(trn_id,val_id) in enumerate(skf.split(new_train,new_train[self.label])):
            print('train fold ',n)
            trn_x ,trn_y = new_train.loc[trn_id,self.features],new_train.loc[trn_id,self.label]
            val_x ,val_y = new_train.loc[val_id,self.features],new_train.loc[val_id,self.label]

            model = self.models[n]

            name = get_model_name(model)
            if name == 'catboost':
                model.fit(trn_x,trn_y,
                          eval_set=[(trn_x, trn_y), (val_x, val_y)],
                          use_best_model=True,
                          metric_period=500,
                          verbose=True,
                          cat_features=cat_features,)
            else:
                trn_data = lgb.Dataset(trn_x, label=trn_y)
                val_data = lgb.Dataset(val_x, label=val_y)
                model = lgb.train(  model.params,
                                      train_set=trn_data,
                                      valid_sets=[trn_data, val_data],
                                      # init_model=model,
                                      verbose_eval=-1)

            if name == 'lightgbm':
                tt_preds += model.predict(self.test[self.features]) / skf.n_splits
                oof_preds[val_id] = model.predict(val_x)

            elif name == 'catboost':
                tt_preds += model.predict_proba(self.test[self.features])[:,1] / skf.n_splits
                oof_preds[val_id] = model.predict_proba(val_x)[:,1]
        auc_score = roc_auc_score(y_true=new_train[self.label], y_score=oof_preds)
        tpr_score = tpr_weight_funtion(y_true=new_train[self.label], y_predict=oof_preds)
        # acc_score = accuracy_score(y_true=self.train[self.label], y_pred=oof_predictions > 0.5)
        print('global auc :', auc_score)
        print('global tpr :', tpr_score)
        return oof_preds,tt_preds

pm_lgb = pseudo_model(train,test,models,label)
pm_cat = pseudo_model(train1,test1,models1,label)

lgb_preds = pm_lgb.get_model_predict()
cat_preds = pm_cat.get_model_predict()

lgb_pseudo_sample = pm_lgb.fliter_sample(lgb_preds,method='absolute')
cat_pseudo_sample = pm_cat.fliter_sample(cat_preds,method='absolute')

merge_tt = pm_lgb.get_common_sample([cat_pseudo_sample,lgb_pseudo_sample])


## refit model
# lgb_oof,lgb_sub = pm_lgb.refit_model(merge_tt,seed=0)
# cat_oof,cat_sub = pm_cat.refit_model(merge_tt,seed=0)

# today = time.strftime("%Y-%m-%d", time.localtime())[5:]
# # test['label'] = cat_sub
# test['label'] = lgb_sub
# sub_train = test[[ID, label]].copy()
#
# sub_test = pd.read_csv('../data/submit.csv')
# sub = sub_test[[ID]].merge(sub_train, on=ID, how='left')
#
# print('null in sub', '\n', sub.isnull().sum())
# sub.fillna(0, inplace=True)
#
# sub.to_csv(f'../result/sub_{today}.csv', index=False)




