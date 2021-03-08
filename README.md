# Gong-Ji-Jin
公积金： http://data.sd.gov.cn/cmpt/cmptDetail.html?id=26

# 运行环境

平台：windows10

语言：python 3.7

package:（注意！不同的文件需要不同的运行环境）

    numpy  	    1.19.5
    
    pandas 	    1.2.1或1.0.5版本，详情见后面的运行环境
    
    catboost 	0.23.2版本
  
    lightgbm 	3.1.1或2.3.0版本，详情见后面的运行环境
  
    tqdm  		4.56.0
  
    sklearn 	0.24.1
  
文件路径：

    data文件夹：含原始训练数据和生成的中间数据。
  
    program文件夹：含所有源程序。
  
    result文件夹：含四个模型各自的预测结果文件。
  
    merge_result文件夹：含最终的融合结果。
  
文件执行顺序：

    （1）在pandas 1.0.5、lightgbm 3.1.1、catboost 0.23.2的环境下，按下述顺序执行文件。

        gen_loan_data.py
  
        lightgbm_model_A.py
  
        catboost_model.py
  
        上述文件执行后，会在data文件夹中生成DK_data_wc.csv文件，在result文件夹中生成lgb_1_result.csv和catboost_result.csv。

    （2）在pandas 1.2.1、lightgbm 2.3.0、xgboost 1.3.3的环境下，按下述顺序执行文件。
    
        lightgbm_model_B.py
    
        xgboost_model.py
    
        上述文件执行后，会在result文件夹中生成lgb_2_result.csv和xgb_result.csv。

    （3）在pandas 1.2.1的环境下，执行下述文件，对四个模型结果进行融合。
    
        merge_result.py
    
        执行后，会在merge_result文件夹生成B榜提交的两个文件中高分的那个文件。

