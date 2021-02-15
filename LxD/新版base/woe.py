import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency


def load_data(data_path='../data/'):
    """加载数据"""

    data_train = pd.read_csv(data_path + 'train.csv')
    data_test = pd.read_csv(data_path + 'test.csv')
    submit = pd.read_csv(data_path + 'submit.csv')

    return data_train, data_test, submit


def graphforbestbin(DF, X, Y, n=5, q=20, graph=True):
    '''
    自动最优分箱函数，基于卡方检验的分箱

    参数：
    DF: 需要输入的数据
    X: 需要分箱的列名
    Y: 分箱数据对应的标签 Y 列名
    n: 保留分箱个数
    q: 初始分箱的个数
    graph: 是否要画出IV图像

    区间为前开后闭 (]

    '''

    DF = DF[[X, Y]].copy()

    DF["qcut"], bins = pd.qcut(DF[X], retbins=True, q=q, duplicates="drop")
    coount_y0 = DF.loc[DF[Y] == 0].groupby(by="qcut").count()[Y]
    coount_y1 = DF.loc[DF[Y] == 1].groupby(by="qcut").count()[Y]
    num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]

    for i in range(q):
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2] + num_bins[1][2],
                num_bins[0][3] + num_bins[1][3])]
            continue

        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i - 1:i + 1] = [(
                    num_bins[i - 1][0],
                    num_bins[i][1],
                    num_bins[i - 1][2] + num_bins[i][2],
                    num_bins[i - 1][3] + num_bins[i][3])]
                break
        else:
            break

    def get_woe(num_bins):
        columns = ["min", "max", "count_0", "count_1"]
        df = pd.DataFrame(num_bins, columns=columns)
        df["total"] = df.count_0 + df.count_1
        df["percentage"] = df.total / df.total.sum()
        df["bad_rate"] = df.count_1 / df.total
        df["good%"] = df.count_0 / df.count_0.sum()
        df["bad%"] = df.count_1 / df.count_1.sum()
        df["woe"] = np.log(df["good%"] / df["bad%"])
        return df

    def get_iv(df):
        rate = df["good%"] - df["bad%"]
        iv = np.sum(rate * df.woe)
        return iv

    IV = []
    axisx = []
    while len(num_bins) > n:
        pvs = []
        for i in range(len(num_bins) - 1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i + 1][2:]
            pv = chi2_contingency([x1, x2])[1]
            pvs.append(pv)

        i = pvs.index(max(pvs))
        num_bins[i:i + 2] = [(
            num_bins[i][0],
            num_bins[i + 1][1],
            num_bins[i][2] + num_bins[i + 1][2],
            num_bins[i][3] + num_bins[i + 1][3])]

        bins_df = pd.DataFrame(get_woe(num_bins))
        axisx.append(len(num_bins))
        IV.append(get_iv(bins_df))

    if graph:
        plt.figure()
        plt.plot(axisx, IV)
        plt.xticks(axisx)
        plt.xlabel("number of box")
        plt.ylabel("IV")
        plt.title(X)
        plt.show()
    return bins_df


def get_woe(df, col, y, bins):
    df = df[[col,y]].copy()
    df["cut"] = pd.cut(df[col],bins)
    bins_df = df.groupby("cut")[y].value_counts().unstack()
    woe = bins_df["woe"] = np.log((bins_df[0]/bins_df[0].sum())/(bins_df[1]/bins_df[1].sum()))
    return woe


def get_cate_woe(df, col, y):
    bins_df = df.groupby(col)[y].value_counts().unstack()
    woe = bins_df['woe'] = np.log((bins_df[0]/bins_df[0].sum())/(bins_df[1]/bins_df[1].sum()))
    return woe


class Woe(object):

    def __init__(self, train, test, auto_col_bins={}, cat_woe_cols=[]):
        """

        :param train: train df
        :param test:  test df
        :param auto_col_bins:  dict，eg: {"GRJCJS":5,}
        :param cat_woe_cols:   list，eg: ['GRZHZT]
        """
        self.train = train
        self.test = test
        self.model_data = train.copy()
        self.auto_col_bins = auto_col_bins
        self.cat_woe_cols = cat_woe_cols

    def plot_iv(self, numerical_cal, n=2, q=20):
        """

        :param numerical_cal: list, [col]
        :param n:
        :param q: number of boxes
        :return:
        """
        for i in numerical_cal:
            print(i)
            graphforbestbin(self.model_data, i, "label", n=n, q=q)

    def get_woe(self):

        # 针对数值特征，返回分箱边界列表bins，记录在字典bins_of_col中
        bins_of_col = {}
        for col in self.auto_col_bins:
            bins_df = graphforbestbin(self.model_data, col,
                                      "label",
                                      n=self.auto_col_bins[col],
                                      q=20,
                                      graph=False)
            bins_list = sorted(set(bins_df["min"]).union(bins_df["max"]))
            # 保证区间覆盖使用 np.inf 替换最大值 -np.inf 替换最小值
            bins_list[0], bins_list[-1] = -np.inf, np.inf
            bins_of_col[col] = bins_list

        woeall = {}
        model_woe = pd.DataFrame(index=self.model_data.index)
        # 对数值特征做woe
        for col in bins_of_col:
            woeall[col] = get_woe(self.model_data, col, "label", bins_of_col[col])
            model_woe[col] = pd.cut(self.model_data[col], bins_of_col[col]).map(woeall[col])
        # 对类别特征做woe
        for col in self.cat_woe_cols:
            woeall[col] = get_cate_woe(self.model_data, col, "label")
            model_woe[col] = self.model_data[col].map(woeall[col])

        # train, test加入woe特征
        # 数值
        for col in bins_of_col:
            self.train['vi' + col] = pd.cut(self.train[col], bins_of_col[col]).map(woeall[col])
            self.test['vi' + col] = pd.cut(self.test[col], bins_of_col[col]).map(woeall[col])

            self.train['vi' + col] = self.train['vi' + col].astype("float64")
            self.test['vi' + col] = self.test['vi' + col].astype("float64")

        # 类别
        for col in self.cat_woe_cols:
            self.train['vi' + col] = self.train[col].map(woeall[col])
            self.test['vi' + col] = self.test[col].map(woeall[col])

            self.train['vi' + col] = self.train['vi' + col].astype("float64")
            self.test['vi' + col] = self.test['vi' + col].astype("float64")

        return self.train, self.test


if __name__ == '__main__':

    trn, tst, sub = load_data()

    auto_col_bins = {'GRJCJS': 5}       # 定义 woe编码目标及分桶数量
    woe = Woe(trn, tst, auto_col_bins=auto_col_bins, cat_woe_cols=[])

    draw = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE']
    woe.plot_iv(draw)

    trn, tst = woe.get_woe()

