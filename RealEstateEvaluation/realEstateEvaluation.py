import re
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import preprocessing

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/submission.csv')

dirpath = os.getcwd()
print(dirpath)

# 觀察單一 column 的資料數量，訓練資料有 69170 個 row，若某些 columns 資料少於 69170 筆，則可能包含 "空值"
print(train.info())
# 確認資料
print(train['index'].head())
print(train['非都市土地使用編定'].head())
print(train['非都市土地使用分區'].head())
print(train['編號'].head())

# Drop 不必要欄位(無法作為特徵使用 或 資訊量過少)
drop_list = ['index', '非都市土地使用編定', '非都市土地使用分區', '編號']
train.drop(drop_list, axis=1, inplace=True)
test.drop(drop_list, axis=1, inplace=True)

# check columns
print(train.info())


# 篩出"數字"特徵，並處理 NaN
numeric_cols = [col for col in train.columns if train[col].dtype in [np.float64, np.int64]]


# 篩出有 NaN 的 column name
col_has_NaN = []
for col in numeric_cols:
    if train[col].isnull().any() == True:
        col_has_NaN.append(col)

print(col_has_NaN)

# 定義填補 NaN 的function
def fill_NaN(col):
    train[col] = train.groupby('鄉鎮市區')[col].transform(lambda x: x.fillna(x.mean()))
    test[col] = test.groupby('鄉鎮市區')[col].transform(lambda x: x.fillna(x.mean()))

# 填補空值
for col in col_has_NaN:
    fill_NaN(col)
    print(col, train[col].isnull().any()) # check 是否有 NaN

# 處理交易年分
print(train['交易年月日'].head()) # 確認資料格式

# 抽取出交易"年分"
def extract_transaction_year(x):
    year = x.astype(str).apply(lambda y: int(y[1:4]))
    return year
    
train['transaction_year_feature'] = extract_transaction_year(train['交易年月日'])
test['transaction_year_feature'] = extract_transaction_year(test['交易年月日'])

print(train.count()) # 原始數據是 69170 筆

# 檢視交易年份分布
train['transaction_year_feature'].plot('hist')
plt.savefig('transaction_year_feature_before.png')
plt.clf()

# 篩掉離群值(年代久遠的紀錄)
train = train[train['transaction_year_feature']>100]
train['transaction_year_feature'].plot('hist')
plt.savefig('transaction_year_feature_after.png')

# 文字特徵 轉 oneHot
from sklearn.preprocessing import OneHotEncoder

def cat_it(cln):
    a = train[cln].astype(str).values
    b = test[cln].astype(str).values
    categories = sorted(list(set(a)))
    enc = OneHotEncoder(categories=[categories], handle_unknown='ignore')
    train_onehot = enc.fit_transform(a.reshape(-1,1)).toarray()
    test_onehot = enc.fit_transform(b.reshape(-1,1)).toarray()
    return train_onehot, test_onehot

material_onehot, test_material_onehot = cat_it('主要建材')
usage_onehot, test_usage_onehot = cat_it('主要用途')
transaction_onehot, test_transaction_onehot = cat_it('交易標的')
building_type_onehot, test_building_type_onehot = cat_it('建物型態')
land_usage_cat_onehot, test_land_usage_cat_onehot = cat_it('都市土地使用分區')
dist_onehot, test_dist_onehot = cat_it('鄉鎮市區')
train_station_onehot, test_train_station_onehot = cat_it('nearest_tarin_station')
location_type_onehot, test_location_type_onehot = cat_it('location_type')
park_type_onehot, test_park_type_onehot = cat_it('車位類別')

# 將 "有" & "無" 數據轉為 1 & 0
train['managed'] = train['有無管理組織'].apply(lambda x: 1 if x == '有' else 0)
test['managed'] = test['有無管理組織'].apply(lambda x: 1 if x == '有' else 0)

train['layout'] = train['建物現況格局-隔間'].apply(lambda x: 1 if x == '有' else 0)
test['layout'] = test['建物現況格局-隔間'].apply(lambda x: 1 if x == '有' else 0)

# 拆解交易筆棟數(將土地、建物、車位等數據分離)
train['land'] = train['交易筆棟數'].apply(lambda s: int(re.findall(r'土地(\d)', s)[0]))
train['build'] = train['交易筆棟數'].apply(lambda s: int(re.findall(r'建物(\d)', s)[0]))
train['park'] = train['交易筆棟數'].apply(lambda s: int(re.findall(r'車位(\d)', s)[0]))

test['land'] = test['交易筆棟數'].apply(lambda s: int(re.findall(r'土地(\d)', s)[0]))
test['build'] = test['交易筆棟數'].apply(lambda s: int(re.findall(r'建物(\d)', s)[0]))
test['park'] = test['交易筆棟數'].apply(lambda s: int(re.findall(r'車位(\d)', s)[0]))

# Predict
from  sklearn.model_selection import train_test_split

# 抽出數字DATA，將預測結果的column移除
X_columns = [col for col in train.columns if train[col].dtype in [np.float64, np.int64]]
X_columns.remove('price_per_ping')
X = train[X_columns].values
print(X_columns) # 確認是否移除 預測結果的column

# 將文字數字串接進 dataframe
X = np.concatenate(
    [X, material_onehot, usage_onehot, transaction_onehot, building_type_onehot, land_usage_cat_onehot, dist_onehot, train_station_onehot, location_type_onehot, park_type_onehot],
    axis=1
)
Y = train[['price_per_ping']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1212)

# 使用 XGBRegressor 演算法 做 回歸預測
import xgboost
from sklearn.metrics import r2_score

# parameters setting
gpuconf = { 
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'max_depth': 10,
    'n_estimators': 1300, #1000
    'learning_rate': 0.02, #0.08
    'gamma': 0,
    'subsample': 0.7,
    'n_jobs': 4,
    'objective': 'reg:squarederror',
    'alpha': 0.00006
}

cpuconf = { 
    'max_depth': 10,
    'n_estimators': 2000,#2000,
    'learning_rate': 0.01,#0.01,
    'gamma': 0,
    'subsample': 0.7,
    'n_jobs': 4,
    'objective': 'reg:squarederror',
    'alpha': 0.00006
}

xgb = xgboost.XGBRegressor(**cpuconf)
xgb.fit(X_train, Y_train)
predictions = xgb.predict(X_test)
# print(f'R2 Score: {r2_score(np.expm1(Y_test), np.expm1(predictions))}')
print(f'R2 Score: {r2_score(Y_test, predictions)}')

X = test[X_columns].values
X = np.concatenate(
    [X, test_material_onehot, test_usage_onehot, test_transaction_onehot, test_building_type_onehot, test_land_usage_cat_onehot, test_dist_onehot, test_train_station_onehot, test_location_type_onehot, test_park_type_onehot],
    axis=1
)

predictions = xgb.predict(X)
my_submission = pd.DataFrame({'index':submission.index,'price_per_ping': predictions})
my_submission.to_csv('submission.csv', index=False)
print(predictions)