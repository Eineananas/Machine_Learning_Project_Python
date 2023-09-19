import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr
import warnings
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pymrmr
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict


warnings.filterwarnings("ignore")

# import data
train_data_cols = [i for i in range(3, 732, 1)]
train_data = pd.read_csv(r'C:/Users/WeiTh/Desktop/Molecular_Descriptor.csv', usecols=train_data_cols, encoding='utf- 8')
train_target = pd.read_csv(r'C:/Users/WeiTh/Desktop/Molecular_Descriptor.csv', usecols=[2], encoding='utf-8')  # 做变量间的分析 离散与离散 连续与连续 离散与连续
train_target = train_target.astype(float)
train_data = train_data.apply(pd.to_numeric, errors='coerce')
train_data = train_data.astype(float)
print(train_data.dtypes)
print(train_target.dtypes)
cols = train_data.columns
col_nm=train_data.columns.to_list()
list_int = []
list_mi=[]
# type of features
print(train_target)
def prs(n):
    x=train_data.iloc[:,n]
    y=train_target.iloc[:,0]
    corr, p_value = pearsonr(x, y)
    corr=np.abs(corr)
    return corr, p_value
    #print("Pearson correlation coefficient:",col_nm[n], corr, p_value)
'''
for i in range(0,729):
    corr, p_value=prs(i)
    list_int.append([col_nm[i], corr, p_value])
    #list_mi.append([col_nm[i],mi])
list_int=pd.DataFrame(list_int)
list_int.to_csv("note.csv")
'''


# corr heat map
list_int.columns = ['Column1', 'Column2', 'Column3']
# Sort in descending order based on the correlation coefficients and select the top 20
top_20_variables = list_int.sort_values(by='Column2', ascending=False).head(20)
corr_matrix = pd.concat([train_target.iloc[:,0],train_data[top_20_variables['Column1']]],axis=1)
print(corr_matrix)
corr_matrix=corr_matrix.corr()
abs_corr_matrix = np.abs(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(abs_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Top 20 Variables')
plt.show()

#Mutual Information
mi_scores = mutual_info_regression(train_data, train_target)
top_features_indices = np.argsort(mi_scores)[-30:]
top_features = train_data.columns[top_features_indices]
top_mi=mi_scores[top_features_indices]
list_mi=[top_features,top_mi,top_features_indices]
list_mi=pd.DataFrame(list_mi)
list_mi.to_csv("note1.csv")

#RFE method for feature selection
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=20)
rfe.fit(train_data, train_target)
rfe_values = rfe.ranking_
feature_rfe = pd.DataFrame({'Feature': train_data.columns, 'RFE Value': rfe_values})
feature_rfe.to_csv("note1.csv")

#Sorting with RF
rf = RandomForestRegressor()
rf.fit(train_data, train_target)
importances = rf.feature_importances_
feature_importances = pd.DataFrame({'Feature': train_data.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
feature_importances.to_csv("note.csv")
print(feature_importances)



list_1=[4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 31, 32, 34, 39, 40, 42, 44, 46, 52, 56, 57, 79, 92, 93, 103, 109, 131, 136, 150, 154, 168, 186, 191, 192, 197, 225, 228, 233, 237, 238, 269, 291, 293, 347, 351, 357, 392, 393, 406, 410, 463, 465, 476, 525, 529, 531, 584, 585, 586, 587, 591, 594, 633, 639, 642, 643, 648, 652, 653, 658, 659, 661, 665, 673, 684, 706, 716, 722, 724, 727]
data1 = pd.concat([train_data.iloc[:, col] for col in list_1], axis=1)
data1.to_csv("note.csv")

#mRMR

def mrmr_feature_selection(X, y, n_features):
    # 将自变量和因变量合并为一个二维列表
    data = pd.concat([X,y],axis=1)
    elected_indices = pymrmr.mRMR(data, "MIQ", n_features)
    # 提取选定的特征
    selected_features = X[elected_indices]
    return selected_features,elected_indices

selected_features, elected_indices= mrmr_feature_selection(data1, train_target, 20)
selected_features=pd.DataFrame(selected_features)
selected_features.to_csv("note.csv")
print(elected_indices)



selected_features = mrmr_feature_selection(train_data, train_target)
feature_importances = pd.DataFrame({'Feature': selected_features, 'Importance': feature_scores_all})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
feature_importances.to_csv("note.csv")
print("Selected Features:", selected_features)
print("Feature Scores:", feature_scores_all)

# final list
list1=[17,13,73,21,66,16,67,72,1,15,4,2,3,22,58,81,27,71,48,64]
td = pd.concat([data1.iloc[:, col] for col in list1], axis=1)
td.to_csv("note.csv")

def pic(df):
    name = df.columns
    for i in range(0,len(name)):
        plt.subplot(5, 4, i + 1)
        # Set the ticks on the x-axis
        #xticks = np.linspace(np.ceil(tb.min()), np.ceil(tb.max()), 11)
        # Set the bins and xticks
        plt.hist(df.iloc[:, i])
        #plt.xticks(rotation=90)
        plt.title(name[i])
    plt.show()
    for i in range(0,len(name)):
        plt.subplot(4, 5, i + 1)
        plt.boxplot(df.iloc[:, i],
                    medianprops={'color': 'green', 'linewidth': '1.5'},
                    meanline=True,
                    showmeans=True,
                    meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                    flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                    )
        plt.title(name[i])
    plt.show()
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 10))
    for i, ax in enumerate(axes.flatten()):
        col_data = df.iloc[:, i]
        stats.probplot(col_data, plot=ax)
        ax.set_title(name[i])
    plt.show()
    print(df.describe())
    df.describe().to_csv("note.csv")
#pic(td)

# Manually remove outliers
td=pd.concat([td,train_target],axis=1)
td = td[td['ATSp4'] < 10000]
td = td[td['ATSc2'] > -1]
td = td[td['McGowan_Volume'] <10]
td = td[td['nBonds2'] < 200]
td = td[td['ETA_Alpha'] < 40]
td = td[td['ATSp2'] < 10000]
td = td[td['ETA_Eta_B'] < 4]
td = td[td['nAtom'] < 300]
td = td[td['ATSp1'] < 10000]
td = td[td['nC'] < 75]
td = td[td['nHeavyAtom'] < 150]
td = td[td['nH'] < 150]
td = td[td['nBondsS2'] < 200]
td = td[td['TopoPSA'] < 200]
td = td[td['SP-5'] < 20]
td = td[td['Kier3'] < 25]
#pic(td)


train_target=td.iloc[:,-1]
td=td.iloc[:, :-1]
#Standard Scaler
scaler = StandardScaler()
td = scaler.fit_transform(td)


#KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_val_score


def knn(k):
    X_train, X_test, y_train, y_test = train_test_split(td, train_target, test_size=0.2, random_state=1)
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_ts=knn.predict(X_train)
    plt.scatter(y_test, y_pred, marker='o', linestyle='-', color='blue')
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    x = [3,4, 5,6,7,8,9,10]
    plt.plot(x, x, 'green', label='y=x')
    plt.legend()
    plt.show()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print("Model generalization ability",[mse,rmse,mae])
    mse1 = mean_squared_error(y_ts, y_train)
    rmse1 = np.sqrt(mean_squared_error(y_ts, y_train))
    mae1 = mean_absolute_error(y_ts, y_train)
    print("Model fitting capability",[mse1,rmse1,mae1])
    return [k, mse, rmse, mae, mse1,rmse1,mae1]


def knn2(k):
    knn = KNeighborsRegressor(n_neighbors=k)
    # 5 folf cross validation
    cv_scores_mse = -cross_val_score(knn, td, train_target, cv=5, scoring='neg_mean_squared_error')
    cv_scores_rmse = np.sqrt(cv_scores_mse)
    cv_scores_mae = -cross_val_score(knn, td, train_target, cv=5, scoring='neg_mean_absolute_error')
    # calculate the mean
    avg_mse = np.mean(cv_scores_mse)
    avg_rmse = np.mean(cv_scores_rmse)
    avg_mae = np.mean(cv_scores_mae)
    # save as matrix
    avg_matrix = np.array([k, avg_mse, avg_rmse, avg_mae])
    print("Average Matrix:",avg_matrix)
    return avg_matrix

#k fold cross validation
def kfold(k,model):
    # Calculate evaluation metrics using cross-validation
    mse_scores = -cross_val_score(model, td, train_target, cv=k, scoring='neg_mean_squared_error')
    mae_scores = -cross_val_score(model, td, train_target, cv=k, scoring='neg_mean_absolute_error')
    mape_scores =-cross_val_score(model, td, train_target, cv=k, scoring='neg_mean_absolute_percentage_error')
    mse_mean = mse_scores.mean()
    mae_mean = mae_scores.mean()
    mape_mean = mape_scores.mean()
    print([mse_scores,mae_scores,mape_scores])
    print(f"MSE: {mse_mean:.4f}")
    print(f"MAE: {mae_mean:.4f}")
    print(f"MAPE: {mape_mean:.4f}%")



knnlist=[]
for i in range(5,50,1):
    ab=[]
    ab=knn2(i)
    knnlist.append(ab)
knnlist=pd.DataFrame(knnlist)
knnlist.to_csv("note.csv")
fig, ax1 = plt.subplots()
# left axis
ax1.plot(knnlist.iloc[:,0], knnlist.iloc[:,1], color='blue', label='MSE')
ax1.plot(knnlist.iloc[:,0], knnlist.iloc[:,2], color='green', label='rMSE')
ax1.set_xlabel('k')
ax1.set_ylabel('MSE,rMSE', color='blue')
# right axis
ax2 = ax1.twinx()
ax2.plot(knnlist.iloc[:,0], knnlist.iloc[:,3], color='red', label='MAE')
ax2.set_ylabel('MAE', color='red')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels)
plt.show()

#knn(22) is the best
X_train, X_test, y_train, y_test = train_test_split(td, train_target, test_size=0.2, random_state=1)
model1=KNeighborsRegressor(n_neighbors=22)
kfold(10,model1)

model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
plt.scatter(y_test, y_pred, marker='o', linestyle='-', color='blue')
plt.xlabel("y_test")
plt.ylabel("y_pred")
x = [3,4,5,6,7,8,9,10]
plt.plot(x, x, 'green', label='y=x')
plt.legend()
plt.show()


#neural network
X_train, X_test, y_train, y_test = train_test_split(td, train_target, test_size=0.2, random_state=1)


# parameters
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (256,)],  # 隐藏层神经元数量
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.0001, 0.001, 0.01]  # 正则化参数
}


model = MLPRegressor(random_state=42)
# search the best hyperparameters using grid search
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)
# print best parameter grid and scores
print('Best Parameters:', grid_search.best_params_)
print('Best Score:', -grid_search.best_score_)

# predict with the best parameter
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_ts=best_model.predict(X_train)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Model generalization ability",[mse,mape,mae])
mse1 = mean_squared_error(y_ts, y_train)
mape1 = mean_absolute_percentage_error(y_ts, y_train)
mae1 = mean_absolute_error(y_ts, y_train)
print("Model fitting capability",[mse1,mape1,mae1])

mlp = MLPRegressor(hidden_layer_sizes=(128,),  # 隐藏层的神经元个数
                   activation='tanh',  # 激活函数
                   alpha=0.0001,#正则化参数
                   solver='adam',  # 优化算法
                   learning_rate='constant',  # 学习率策略
                   )
kfold(10,mlp)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
plt.scatter(y_test, y_pred, marker='o', linestyle='-', color='blue')
plt.xlabel("y_test")
plt.ylabel("y_pred")
x = [3,4,5,6,7,8,9,10]
plt.plot(x, x, 'green', label='y=x')
plt.legend()
plt.show()



X_train, X_test, y_train, y_test = train_test_split(td, train_target, test_size=0.2, random_state=1)

# GBDT
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['auto', 'sqrt']
}

gbdt = GradientBoostingRegressor()
grid_search = GridSearchCV(estimator=gbdt, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best Parameters: ", grid_search.best_params_)
print("Best Score (MSE): ", -grid_search.best_score_)

gbr=GradientBoostingRegressor(
    learning_rate=0.1,         # 学习率，控制每棵树的贡献程度
    n_estimators=100,          # 决策树的数量
    subsample=0.8,             # 训练每棵树时使用的子样本比例
    criterion='friedman_mse',  # 分裂节点的质量评估准则，可选参数：'friedman_mse'（默认）、'mse'或'mae'
    min_samples_split=6,       # 节点分裂所需的最小样本数
    min_samples_leaf=2,        # 叶节点所需的最小样本数
    max_depth=7,               # 决策树的最大深度
    max_features='sqrt',         # 每棵树考虑的最大特征数，可选参数：'auto'（默认）、'sqrt'、'log2'或None
)
gbr.fit(X_train, y_train)


y_pred = gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape=mean_absolute_percentage_error(y_test, y_pred)
print("Test MSE: ", mse)
print("Test MAE: ", mae)
print("Test MAPE: ", mape)
plt.scatter(y_test, y_pred, marker='o', linestyle='-', color='blue')
plt.xlabel("y_test")
plt.ylabel("y_pred")
x = [3,4,5,6,7,8,9,10]
plt.plot(x, x, 'green', label='y=x')
plt.legend()
plt.show()


#GradientBoosting
gbr=GradientBoostingRegressor(
    learning_rate=0.1,         # 学习率，控制每棵树的贡献程度
    n_estimators=100,          # 决策树的数量
    subsample=0.8,             # 训练每棵树时使用的子样本比例
    criterion='friedman_mse',  # 分裂节点的质量评估准则，可选参数：'friedman_mse'（默认）、'mse'或'mae'
    min_samples_split=6,       # 节点分裂所需的最小样本数
    min_samples_leaf=2,        # 叶节点所需的最小样本数
    max_depth=7,               # 决策树的最大深度
    max_features='sqrt',         # 每棵树考虑的最大特征数，可选参数：'auto'（默认）、'sqrt'、'log2'或None
)
kfold(10,gbr)

X_train, X_test, y_train, y_test = train_test_split(td, train_target, test_size=0.2, random_state=1)

param_grid = {
    'learning_rate': [0.01, 0.1, 1.0],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'lambda': [0.1, 1.0, 10.0],
    'alpha': [0.1, 1.0, 10.0]
}

xgb = XGBRegressor()
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Score (MSE): ", -grid_search.best_score_)

xgb= XGBRegressor(
    n_estimators=100,  # 决策树数量
    learning_rate=0.1,  # 学习率
    max_depth=7,  # 决策树最大深度
    subsample=1.0,  # 每个决策树样本的比例
    colsample_bytree=0.8,  # 每个决策树特征的比例
    reg_alpha=1.0,  # L1 正则化系数
    reg_lambda=0.1,  # L2 正则化系数
)

xgb.fit(X_train, y_train)
# cross validation
y_pred = xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape=mean_absolute_percentage_error(y_test, y_pred)
print("Test MSE: ", mse)
print("Test MAE: ", mae)
print("Test MAPE: ", mape)
plt.scatter(y_test, y_pred, marker='o', linestyle='-', color='blue')
plt.xlabel("y_test")
plt.ylabel("y_pred")
x = [3,4,5,6,7,8,9,10]
plt.plot(x, x, 'green', label='y=x')
plt.legend()
plt.show()
xgb= XGBRegressor(
    n_estimators=100,  # 决策树数量
    learning_rate=0.1,  # 学习率
    max_depth=7,  # 决策树最大深度
    subsample=1.0,  # 每个决策树样本的比例
    colsample_bytree=0.8,  # 每个决策树特征的比例
    reg_alpha=1.0,  # L1 正则化系数
    reg_lambda=0.1,  # L2 正则化系数
)
kfold(10,xgb)










