import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler


class K_means(object):

    def __init__(self, train, test, k):
        self.train = train
        self.test = test
        self.k = k

    def pred(self, seed=2021):

        data = pd.concat([self.train, self.test], axis=0).reset_index(drop=True)
        km = KMeans(n_clusters=self.k, random_state=seed)               # 将数据集分为k类
        X = data.drop(['id', 'label'], axis=1).copy()

        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(X.values)

        X = min_max_scaler.transform(X.values)
        y_pre = km.fit_predict(X)
        print("K means Scores: ", calinski_harabasz_score(X, y_pre))    # 分数越高，表示聚类的效果越好

        data['cluster'] = y_pre

        # get train & test data
        train_pre = data[data['label'].isna() == False].reset_index(drop=True)['cluster']
        test_pre = data[data['label'].isna() == True].reset_index(drop=True)['cluster']

        return train_pre, test_pre
