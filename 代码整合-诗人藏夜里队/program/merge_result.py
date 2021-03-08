import pandas as pd


if __name__ == '__main__':

    data_path = '../data/'

    train = pd.read_csv(data_path + 'train.csv')
    test = pd.read_csv(data_path + 'test.csv')
    submit = pd.read_csv(data_path + 'submit.csv')

    zhiwu = list(test[test['ZHIWU'] == 1]['id'].values)

    res1 = pd.read_csv('../result/lgb_1_result.csv')
    res2 = pd.read_csv('../result/catboost_result.csv')
    res3 = pd.read_csv('../result/lgb_2_result.csv')
    res4 = pd.read_csv('../result/xgb_result.csv')

    res1 = res1.rename(columns={'label': 'label_1'})
    res2 = res2.rename(columns={'label': 'label_2'})
    res3 = res3.rename(columns={'label': 'label_3'})
    res4 = res4.rename(columns={'label': 'label_4'})

    res1 = pd.merge(res1, res2, on='id', how='left')
    res1 = pd.merge(res1, res3, on='id', how='left')
    res1 = pd.merge(res1, res4, on='id', how='left')

    res1['label'] = res1.apply(lambda x: ((x['label_3'] + x['label_4']) / 2) if x['id'] in zhiwu else (
                (x['label_1'] + x['label_2'] + x['label_3'] + x['label_4']) / 4), axis=1)

    res1[['id', 'label']].to_csv('../merge_result/final_result_1.csv', index=False)

    print('finish !')

    # ---------- 以下是B榜第二个文件的融合方案，rank融合 --------------

    # res1['label_1_rank'] = res1['label_1'].rank(method='average')
    # res1['label_2_rank'] = res1['label_2'].rank(method='average')
    # res1['label_3_rank'] = res1['label_3'].rank(method='average')
    # res1['label_4_rank'] = res1['label_4'].rank(method='average')

    # res1['label'] = res1.apply(lambda x: np.sqrt(
    #     (x['label_1_rank'] ** 0.5) + (x['label_2_rank'] ** 0.5) + (x['label_3_rank'] ** 0.5) + (
    #                 x['label_4_rank'] ** 0.5)), axis=1)

    # high = max(res1['label'])
    # low = min(res1['label'])
    # res1['label'] = res1['label'].apply(lambda x: (x - low) / (high - low))
    # res1[['id', 'label']].to_csv('../merge_result/rank_merge_result.csv', index=False)


