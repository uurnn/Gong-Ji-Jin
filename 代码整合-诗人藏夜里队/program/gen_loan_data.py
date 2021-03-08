# coding:utf-8
import math
import numpy as np
import pandas as pd
from tqdm import tqdm


def monthlyPayment(principal, year_rate, year_duration):
    monthly_rate = year_rate / (12 * 100)   # convert 4.9 to 0.049 and  monthly interest rate
    month_amounts =  year_duration * 12

    monthly_paied = []
    surplus_list = []
    for i in range (1, month_amounts + 1):
        #每月应还利息
        # monthly_interest_payable = principal * monthly_rate * ((1 + monthly_rate) ** month_amounts - (1 + monthly_rate) ** (i - 1 ))/ ((1 + monthly_rate) ** month_amounts -1)
        #每月应还本金
        monthly_principal_payable = principal * monthly_rate * (1 + monthly_rate) ** (i - 1)/ ((1 + monthly_rate) ** month_amounts -1)
        monthly_paied.append(monthly_principal_payable)
        #当月剩余本金
        monthly_surplus = principal - math.fsum(monthly_paied)
        surplus_list.append(monthly_surplus)
    return surplus_list
def deal_dkll(train,test):
    ##只能存在四种利率：2.708（3.25），2.979（3.575），2.292，2.521
    for df in [train,test]:
        df['DKLL'] = np.round(df['DKLL'],3)
        year5_M_rate_mask = df['DKLL'].isin([3.25,3.025])
        year1_M_rate_mask = df['DKLL'].isin([2.750])
        year5_M2_rate_mask = df['DKLL'].isin([3.575])
        df.loc[year5_M_rate_mask,'DKLL'] = 2.708
        df.loc[year1_M_rate_mask,'DKLL'] = 2.292
        df.loc[year5_M2_rate_mask,'DKLL'] = 2.979

        double_house_mask = df['DKLL'].isin([2.979,2.5212])
        df['DKLL_check'] = (year5_M_rate_mask|year5_M2_rate_mask|year1_M_rate_mask).astype(int)
        df['double_house'] = (double_house_mask).astype(int)

    return ['DKLL_check','double_house']


def find_DKNX(ffe,ye,ll):
    """
    我们需要找到1.贷款年限 2.贷款第几期
    :param ffe:
    :param ye:
    :param ll:
    :return:
    """
    # if ffe == ye:
    #     return -1,-1
    if ll == 2.708:
        tt_dict = loan_long_dict.copy()
    elif ll == 2.979:
        tt_dict = loan_double_long_dict.copy()
    elif ll == 2.292:
        tt_dict = loan_short_dict.copy()
    elif ll == 2.521:
        tt_dict = loan_doubel_short_dict.copy()
    else:
        print('暂不支持该利率')
        return [np.nan,np.nan]
    diff_min = np.inf
    n_min = [0,0]
    act_ye =  0
    for n in tt_dict:
        pay_list = [abs(ye-x*ffe*(10e-6)) for x in tt_dict[n]]
        _ = np.argmin(pay_list)
        tmp_min = pay_list[_]
        if tmp_min < diff_min:
            diff_min = tmp_min
            n_min = [n,_+1]
            act_ye = pay_list[_]
    return n_min,act_ye

if __name__ == '__main__':
    loan_long_dict = {}
    for year in range(6,31):
        principal = 100000
        year_rate = 3.25
        year_duration = year
        loan_long_dict[year] = monthlyPayment(principal, year_rate, year_duration)

    loan_double_long_dict = {}
    for year in range(6,31):
        principal = 100000
        year_rate = 3.575
        year_duration = year
        loan_double_long_dict[year] = monthlyPayment(principal, year_rate, year_duration)


    loan_short_dict ={}
    for year in range(1,6):
        principal = 100000
        year_rate = 2.75
        year_duration = year
        loan_short_dict[year] = monthlyPayment(principal, year_rate, year_duration)

    loan_doubel_short_dict ={}
    for year in range(1,6):
        principal = 100000
        year_rate = 3.025
        year_duration = year
        loan_doubel_short_dict[year] = monthlyPayment(principal, year_rate, year_duration)
    # ffe = 175237.0
    # ye = 20653.755
    # ll = 2.292
    # a = find_DKNX(ffe, ye, ll)

    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    drop_zhiwu = True
    if drop_zhiwu:
        # train = train[train['ZHIWU'] == 0].reset_index(drop=True)
        test = test[test['ZHIWU'] == 0].reset_index(drop=True)

    check_features = deal_dkll(train, test)
    data = pd.concat([train,test],axis=0,ignore_index=True)
    data = data[['DKFFE','DKYE','DKLL']]
    data[['DKFFE','DKYE']] = data[['DKFFE','DKYE']].apply(lambda x: x - 237)
    data.drop_duplicates(subset=['DKFFE','DKYE','DKLL'],keep='last',inplace=True,ignore_index=True)

    res_time = []
    res_tt = []
    res_now = []
    for n,row in tqdm(data.iterrows()):
        ffe = row['DKFFE']
        ye = row['DKYE']
        ll = row['DKLL']
        print(ffe, ye, ll)
        if ffe == ye :
            res_time.append(-1)
            res_now.append(-1)
            res_tt.append(-1)
        else:
            tmp_time,act_ye = find_DKNX(ffe,ye,ll)
            print(ffe,ye,act_ye*ffe*(10e-6),ll,tmp_time[0])
            res_time.append(tmp_time[0])
            res_now.append(tmp_time[1])
            res_tt.append(act_ye)
    data['DKNX'] = res_time
    data['DKQS'] = res_now
    data['DKWC'] = res_tt
    data.to_csv('../data/DK_data_wc_v1.csv', index=False)
