from sklearn.model_selection import *
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import random
import os
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import lightgbm as lgb
import warnings

from suanfa import Tradaboost

warnings.filterwarnings('ignore')
import os
import gc

train_A = pd.read_csv('C:/Users/ASUS/Desktop/pythonProject2/A_train.csv')

train_B = pd.read_csv('C:/Users/ASUS/Desktop/pythonProject2/B_train.csv')
test = pd.read_csv('C:/Users/ASUS/Desktop/pythonProject2/B_test.csv')
sample = pd.read_csv('C:/Users/ASUS/Desktop/pythonProject2/submit_sample.csv')

#数据的处理---------------------------------


train_B_info = train_B.describe()
useful_col = []
for col in train_B_info.columns:
    #if train_B_info.ix[0,col] > train_B.shape[0]*0.01:
    print('ok')
    train_B_1 = train_B.dropna(axis=1,thresh=40)
    useful_col.append(col)

"""
# -----------检查数据中是否有缺失值，以下两种方式均可
# Flase:对应特征的特征值中无缺失值
# True：有缺失值
#print(train_df.isnull().any())
#print(np.isnan(train_df).any())

# 查看缺失值记录
train_null = pd.isnull(train_df)
#train_null = train_df[train_null == True]
#print(train_null)
# 缺失值处理，以下两种方式均可
# 删除包含缺失值的行
#train_df.dropna(inplace=True)
# 缺失值填充
train_df.fillna('-999')
#train_df['UserInfo_170'].unique()
#print ('填充后查看------------------------------')
#print(train_df.isnull().any())
train_null = pd.isnull(train_df)
#print(train_null)

# -----------检查是否包含无穷数据
# False:包含
# True:不包含
#print(np.isfinite(train_df).all())
# False:不包含
# True:包含
#print(np.isinf(train_df).all())

# 数据处理
train_inf = np.isinf(train_df)
train_df[train_inf] = 0

#-----------------------------------高线性相关处理
relation = train_df.corr()
length = relation.shape[0]
high_corr = list()
final_cols = []
del_cols =[]
for i in range(length):
    if relation.columns[i] not in del_cols:
        final_cols.append(relation.columns[i])
        for j in range(i+1,length):
            if (relation.iloc[i,j] > 0.98) and (relation.columns[j] not in del_cols):
                del_cols.append(relation.columns[j])
train_df = train_df[final_cols]
"""

train_B_1 = train_B[useful_col].copy()
#train_B_1 = train_B_1.fillna(-999)
relation = train_B_1.corr()

train_A_1 = train_A[useful_col].copy()
#train_A_1 = train_A_1.fillna(-999)
length = relation.shape[0]
high_corr = list()
final_cols = []
del_cols = []
for i in range(length):
    if relation.columns[i] not in del_cols:
        final_cols.append(relation.columns[i])
        for j in range(i+1, length):
            if (relation.iloc[i,j] > 0.98) and (relation.columns[j] not in del_cols):
                del_cols.append(relation.columns[j])

train_B_1 = train_B_1[final_cols]
train_A_1 = train_A_1[final_cols]

train_B_flag = train_B_1['flag']
train_B_1.drop('no', axis = 1, inplace = True)
train_B_1.drop('flag', axis = 1, inplace = True)

train_A_flag = train_A_1['flag']
train_A_1.drop('no', axis = 1, inplace = True)
train_A_1.drop('flag', axis = 1, inplace = True)
print('ok')

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(456)

#  -------xgboost------------
train_B_1_valid,train_B_1_test,train_B_1_valid_y,train_B_1_test_y=train_test_split(train_B_1,train_B_flag,test_size=0.5)

#dtrain_B = xgb.DMatrix(data = train_B_1, label = train_B_flag)
Trate = 0.25
params = {'booster':'gbtree',
          'eta':0.1,
          'max_depth':20,
          'max_delta_step':0,
          'subsample':1,
          'colsample_bytree':1,
          'base_score':Trate,
          'objective':'binary:logistic',
          'lambda':5,
          'alpha':8,
          'n_estimators':500,
          'random_seed':100,
          'n_jobs':-1
}
clf=xgb.XGBClassifier(**params)
clf.fit(train_A_1,train_A_flag)

print ('ok')
y_pred_A=clf.predict_proba(train_A_1)[:,1]
y_pred_B_valid=clf.predict_proba(train_B_1_valid)[:,1]
y_pred_B_test=clf.predict_proba(train_B_1_test)[:,1]

print(f" train AUC = {roc_auc_score(train_A_flag,y_pred_A)}")
print(f" valid AUC = {roc_auc_score(train_B_1_valid_y,y_pred_B_valid)}")
print(f" test AUC = {roc_auc_score(train_B_1_test_y,y_pred_B_test)}")
#--------------------------------------------------------

#-----------------Tradaboost-------------------------------------
clf=Tradaboost(N=200,base_estimator=xgb.XGBClassifier(**params), \
threshold=0.92975,score=roc_auc_score)
clf.fit(train_A_1.values,train_B_1_valid.values,train_A_flag,train_B_1_valid_y.values,50)
#---------------------------------------------------------------------

for i,estimator in enumerate(clf.estimators):
    print('迭代的次数：'+str(i+1)+' estimator:')
    y_pred_A=estimator.predict_proba(train_A_1.values)[:,1]
    y_pred_B_valid=estimator.predict_proba(train_B_1_valid.values)[:,1]
    y_pred_B_test=estimator.predict_proba(train_B_1_test.values)[:,1]

    print(f" train AUC = {roc_auc_score(train_A_flag,y_pred_A)}")
    print(f" valid AUC = {roc_auc_score(train_B_1_valid_y,y_pred_B_valid)}")
    print(f" test AUC = {roc_auc_score(train_B_1_test_y,y_pred_B_test)}")
    print('\n')
    print('**********************')

#---------------------------------------------------------------------