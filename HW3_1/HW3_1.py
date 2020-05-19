import sys
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
from mplfinance.original_flavor import plot_day_summary_ohlc
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn import preprocessing, linear_model

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor



train = pd.read_csv('source/Train.csv')
test = pd.read_csv('source/Test.csv')
print(len(test))

train['dataclass'] = 0
test['dataclass'] = 1

full = pd.concat([train, test])

# 與前一天close price比較 漲:1 跌:0
full['day_movement'] = np.where(full['Close Price'].shift(-1) > full['Close Price'], 1, 0)


# 當天漲跌，ohlc chart用
full['movement'] = full['Close Price'] - full['Open Price']
full.loc[ full['movement'] <= 0, 'movement'] = 0
full.loc[ full['movement'] > 0, 'movement'] = 1

# 轉換日期時間，繪製chart用
full.index = pd.to_datetime(full['Date'])


# 轉換成list of lists資料，ohlc chart用
dvalues = full[['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume']].values.tolist()

# 將時間日期轉換成Matplotlib date format
pdates = mdates.date2num(full.index)

# list of lists型態ohlc資料，每條list型態為[date, open, high, low, close]
ohlc = [ [pdates[i]] + dvalues[i] for i in range(len(pdates)) ]


# 繪製ohlc candle stick chart
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (12,6))

# 取其中200筆資料繪製
plot_day_summary_ohlc(ax, ohlc[-200:], ticksize = 5, colorup = 'red', colordown = 'green')

ax.set_xlabel('Date')
ax.set_ylabel('Price ')
ax.set_title('S&P 500 OHLC candle stick chart')

# dates資料轉換成字串顯示
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# shift y-limits of the candlestick留空間給volume bar chart
pad = 0.5
yl = ax.get_ylim()
ax.set_ylim(yl[0]-(yl[1]-yl[0])*pad,yl[1])

ax.grid(True)

# volume bar chart
ax2 = ax.twinx()

bar_train = full[-200:]
# 收盤價與開盤價比為負
neg = bar_train.loc[bar_train['movement'] == 0, 'Date']
negv = bar_train.loc[bar_train['movement'] == 0, 'Volume']
# 收盤價與開盤價比為正
pos = bar_train.loc[bar_train['movement'] == 1, 'Date']
posv = bar_train.loc[bar_train['movement'] == 1, 'Volume']
neg = pd.to_datetime(neg)
pos = pd.to_datetime(pos)
ax2.bar(neg, negv, color='green', width=0.5, align='center' , alpha=0.5)
ax2.bar(pos, posv, color='red', width=0.5, align='center' , alpha=0.5)


ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Volume ')

# y坐標軸值限制
ax2.set_ylim(0, max(train['Volume']))
# y坐標軸刻度
yticks = ax2.get_yticks()
ax2.set_yticks(yticks[::3])

ax2.grid(False)

# 計算SMA、EMA
hsma40 = bar_train['High Price'].rolling(5).mean()
lsma40 = bar_train['Low Price'].rolling(30).mean()
ema5 = bar_train['Close Price'].ewm(5).mean()
ema15 = bar_train['Close Price'].ewm(15).mean()

ax.plot(hsma40, color = 'blue', linewidth = 2, label='High, 40-Day SMA')
ax.plot(lsma40, color = 'blue', linewidth = 2, label='Low, 40-Day SMA')
ax.plot(ema15, color = 'red', linestyle='--', linewidth = 2, label='Close, 15-Day EMA')
ax.plot(ema5, color = 'green', linestyle='--', linewidth = 2, label='Close, 5-Day EMA')



# feature generation
# 5 day simple moving average
full['avg_price_5'] = full['Close Price'].rolling(5).mean()
# 5 day simple moving average movement
full['avg_movement'] = np.where(full['avg_price_5'].shift(-5) > full['avg_price_5'], 1, 0)
# 30 day simple moving average
full['avg_price_30'] = full['Close Price'].rolling(30).mean()
# 30 day simple moving average movement
full['30avg_movement'] = np.where(full['avg_price_30'].shift(-30) > full['avg_price_30'], 1, 0)


# 5 day Exponential Moving Average 
full['ema5'] = full['Close Price'].ewm(5).mean()
# ema與sma相較，對近期價格影響較大，所以只比較前1天
# 5 day Exponential Moving Average movement
full['ema5_movement'] = np.where(full['ema5'].shift(-1) > full['ema5'], 1, 0)
# 30 day Exponential Moving Average 
full['ema30'] = full['Close Price'].ewm(30).mean()
# 30 day Exponential Moving Average  movement
full['ema30_movement'] = np.where(full['ema30'].shift(-1) > full['ema30'], 1, 0)

#volume與前一天比較 
full['volume_movement'] = np.where(full['Volume'].shift(-1) > full['Volume'], 1, 0)

# 只留下movement值，降低複雜度
full = full.drop(['avg_price_30', 'avg_price_5', 'movement', 'ema5', 'ema30', 'Volume'], axis = 1)

# 分開train、test資料
X_train = full[full['dataclass'] == 0]
X_test = full[full['dataclass'] == 1]

# 捨棄Nan值
X_train = X_train.dropna()
X_train = X_train.reset_index(drop=True)
X_test = X_test.dropna()
X_test2 = X_test.reset_index(drop=True)

# 捨棄無關資料
x_train = X_train.drop(['Date', 'dataclass', 'Open Price', 'Close Price', 'High Price', 'Low Price'], axis = 1)

# 與目標值'day_movement'相關程度
sorted_corrs = x_train.corr()['day_movement'].sort_values()

print(sorted_corrs)
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(x_train[sorted_corrs.index].corr())
# plt.show()



# 切取訓練集、驗證集
X_train, X_test, y_train, y_test = train_test_split(x_train.drop('day_movement', axis = 1),
                                                    x_train['day_movement'], test_size = 0.2, 
                                                    random_state = 2)
# 亂數種子
random_seed = 4

# random_forest = RandomForestClassifier()
# random_grid = {'bootstrap': [True, False],
#                'max_depth': [50, 100, 150, None],
#                'max_features': ['auto', 'sqrt', 'log2'],
#                'min_samples_leaf': [1, 2, 4],
#                'min_samples_split': [2, 5, 10],
#                'n_estimators': [50, 100, 300]}

# grid = RandomizedSearchCV(random_forest,random_grid,cv = 3,scoring = 'neg_log_loss',n_iter=300,n_jobs = -1)
# #在训练集上训练
# grid.fit(X_train, y_train)
# #返回最优的训练器
# random_forest = grid.best_estimator_
# print(random_forest)
# #输出最优训练器的精度
# print(grid.best_score_)



kf = KFold(n_splits=5,                                # 設定 K 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(X_train)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(X_train):    # 每個迴圈都會產生不同部份的資料
    train_x_split = X_train.iloc[train_index]         # 產生訓練資料
    train_y_split = y_train.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = X_train.iloc[valid_index]         # 產生驗證資料
    valid_y_split = y_train.iloc[valid_index]         # 產生驗證資料標籤
    
   
    random_forest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=100, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=random_seed,
                       verbose=0, warm_start=False)
    random_forest.fit(train_x_split, train_y_split)

    train_pred_y = random_forest.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = random_forest.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
        'average train accuracy: {}\n' +
        '    min train accuracy: {}\n' +
        '    max train accuracy: {}\n' +
        'average valid accuracy: {}\n' +
        '    min valid accuracy: {}\n' +
        '    max valid accuracy: {}').format(
        np.mean(train_acc_list),                          # 輸出平均訓練準確度
        np.min(train_acc_list),                           # 輸出最低訓練準確度
        np.max(train_acc_list),                           # 輸出最高訓練準確度
        np.mean(valid_acc_list),                          # 輸出平均驗證準確度
        np.min(valid_acc_list),                           # 輸出最低驗證準確度
        np.max(valid_acc_list)                            # 輸出最高驗證準確度
    ))

test_pred_y = random_forest.predict(X_test2.drop(['Date','day_movement', 'dataclass', 'Open Price', 'Close Price', 'High Price', 'Low Price'], axis = 1))
print(len(test_pred_y))

count = 0
for index in range(len(test_pred_y)):
    if ((test_pred_y[index] - X_test2['day_movement'][index]) == 0):
        count += 1

print(count / len(X_test2))


# logistic_regr = linear_model.LogisticRegression()
# logistic_regr_grid = {'C': [0.1, 0.3, 0.5, 0.9, 1],
#                     'class_weight': ['balanced', None],
#                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#                     'max_iter': [50, 100, 200, 500, 1000]}

# grid = RandomizedSearchCV(logistic_regr,logistic_regr_grid,cv = 3,scoring = 'neg_log_loss',n_jobs = -1)
# #在训练集上训练
# grid.fit(train_x_split, train_y_split)
# #返回最优的训练器
# logistic_regr = grid.best_estimator_
# # print(best_logistic_regr)
# # #输出最优训练器的精度
# # print(grid.best_score_)

kf = KFold(n_splits=5,                                # 設定 K 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(X_train)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(X_train):    # 每個迴圈都會產生不同部份的資料
    train_x_split = X_train.iloc[train_index]         # 產生訓練資料
    train_y_split = y_train.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = X_train.iloc[valid_index]         # 產生驗證資料
    valid_y_split = y_train.iloc[valid_index]         # 產生驗證資料標籤
    
   
    logistic_regr = linear_model.LogisticRegression(C=0.3, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=50,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=random_seed, solver='sag', tol=0.0001, verbose=0,
                   warm_start=False)
    logistic_regr.fit(train_x_split, train_y_split)


    train_pred_y = logistic_regr.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = logistic_regr.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
        'average train accuracy: {}\n' +
        '    min train accuracy: {}\n' +
        '    max train accuracy: {}\n' +
        'average valid accuracy: {}\n' +
        '    min valid accuracy: {}\n' +
        '    max valid accuracy: {}').format(
        np.mean(train_acc_list),                          # 輸出平均訓練準確度
        np.min(train_acc_list),                           # 輸出最低訓練準確度
        np.max(train_acc_list),                           # 輸出最高訓練準確度
        np.mean(valid_acc_list),                          # 輸出平均驗證準確度
        np.min(valid_acc_list),                           # 輸出最低驗證準確度
        np.max(valid_acc_list)                            # 輸出最高驗證準確度
    ))

test_pred_y = logistic_regr.predict(X_test2.drop(['Date','day_movement', 'dataclass', 'Open Price', 'Close Price', 'High Price', 'Low Price'], axis = 1))
print(len(test_pred_y))

count = 0
for index in range(len(test_pred_y)):
    if ((test_pred_y[index] - X_test2['day_movement'][index]) == 0):
        count += 1

print(count / len(X_test2))


# mlp = MLPClassifier()
# mlp_grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#             'alpha': [0.0001, 0.05],
#             'learning_rate': ['constant','adaptive'],
#             'max_iter': [200, 500, 700]}

# grid = RandomizedSearchCV(mlp,mlp_grid,cv = 3,scoring = 'neg_log_loss',n_iter=300,n_jobs = -1)
# #在训练集上训练
# grid.fit(train_x_split, train_y_split)
# #返回最优的训练器
# mlp = grid.best_estimator_
# # print(best_mlp)
# # #输出最优训练器的精度
# # print(grid.best_score_)

kf = KFold(n_splits=5,                                # 設定 K 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(X_train)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(X_train):    # 每個迴圈都會產生不同部份的資料
    train_x_split = X_train.iloc[train_index]         # 產生訓練資料
    train_y_split = y_train.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = X_train.iloc[valid_index]         # 產生驗證資料
    valid_y_split = y_train.iloc[valid_index]         # 產生驗證資料標籤
    
    mlp = MLPClassifier(activation='relu', alpha=0.05, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=500,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=random_seed, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
    mlp.fit(train_x_split,train_y_split)


    train_pred_y = mlp.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = mlp.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
        'average train accuracy: {}\n' +
        '    min train accuracy: {}\n' +
        '    max train accuracy: {}\n' +
        'average valid accuracy: {}\n' +
        '    min valid accuracy: {}\n' +
        '    max valid accuracy: {}').format(
        np.mean(train_acc_list),                          # 輸出平均訓練準確度
        np.min(train_acc_list),                           # 輸出最低訓練準確度
        np.max(train_acc_list),                           # 輸出最高訓練準確度
        np.mean(valid_acc_list),                          # 輸出平均驗證準確度
        np.min(valid_acc_list),                           # 輸出最低驗證準確度
        np.max(valid_acc_list)                            # 輸出最高驗證準確度
    ))

test_pred_y = mlp.predict(X_test2.drop(['Date','day_movement', 'dataclass', 'Open Price', 'Close Price', 'High Price', 'Low Price'], axis = 1))
print(len(test_pred_y))

count = 0
for index in range(len(test_pred_y)):
    if ((test_pred_y[index] - X_test2['day_movement'][index]) == 0):
        count += 1

print(count / len(X_test2))

