import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
from fancyimpute import IterativeImputer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")


#import data from csv
train_data_cols = [i for i in range(1, 27, 1)]  # train_data_cols.append(1)
train_data = pd.read_csv(r'C:/Users/WeiTh/Desktop/data.csv', usecols=train_data_cols, encoding='utf- 8')
train_target = pd.read_csv(r'C:/Users/WeiTh/Desktop/data.csv', usecols=[27], encoding='utf-8')  # 做变量间的分析 离散与离散 连续与连续 离散与连续
train_target = train_target.astype(float)
train_data = train_data.apply(pd.to_numeric, errors='coerce')
train_data = train_data.astype(float)
print(train_data.dtypes)
print(train_target.dtypes)
cols = train_data.columns
col_nm=train_data.columns.to_list()
dt = pd.concat([train_target.iloc[:,0],train_data],axis=1)
def pic(df):
    name = df.columns
    for i in range(0,len(name)):
        plt.subplot(5, 6, i + 1)
        #xticks = np.linspace(np.ceil(tb.min()), np.ceil(tb.max()), 11)
        # set bins and xticks
        plt.hist(df.iloc[:, i])
        #plt.xticks(rotation=90)
        plt.title(name[i])
    plt.show()
    for i in range(0,len(name)):
        plt.subplot(5, 6, i + 1)
        plt.boxplot(df.iloc[:, i],
                    medianprops={'color': 'green', 'linewidth': '1.5'},
                    meanline=True,
                    showmeans=True,
                    meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                    flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                    )
        plt.title(name[i])
    plt.show()
    for i in range(0,len(name)):
        plt.subplot(5, 6, i + 1)
        col_data = df.iloc[:, i]
        stats.probplot(col_data, plot=plt)
        plt.title(name[i])
    plt.show()
    print(df.describe())
    df.describe().to_csv("note.csv")
pic(dt)


data=dt

# Method One: Impute missing data using mean imputation
data_mean = data.fillna(data.mean())
missing_matrix = data.isnull().astype(int)
missing_percentages = dt.isnull().mean()*100
print(missing_percentages)
print(missing_matrix)


def mis1():
    plt.bar(missing_percentages.index, missing_percentages.values)
    plt.title("Missing Percentages")
    plt.xticks(rotation=90)
    plt.show()


def prs(m,n):
    x=data_mean.iloc[:,m]
    y=missing_matrix.iloc[:,n]
    corr, p_value = pearsonr(x, y)
    corr=np.abs(corr)
    return corr, p_value

lprs = pd.DataFrame(1, index=range(len(col_nm)), columns=range(len(col_nm)))
for i in range(0,len(col_nm)):
    for j in range(0,len(col_nm)):
        a,b=prs(i,j)
        lprs.iloc[i,j]=a
print(lprs)

def mis2():
    sns.heatmap(lprs, annot=True, cmap='coolwarm')
    variable_names = missing_matrix.columns
    plt.xticks(range(len(col_nm) + 1), missing_percentages.index, rotation=90)
    plt.yticks(range(len(col_nm) + 1), missing_percentages.index, rotation=0)
    plt.xlabel("Missing_Matrix.iloc[:,i]")
    plt.ylabel("Data.iloc[:,i]")
    plt.show()

mis1()
mis2()

#Method Two: Iterative Impute
imputer = IterativeImputer()
imputed_data = imputer.fit_transform(data)
imputed_df1 = pd.DataFrame(imputed_data, columns=data.columns)

# Cyclically impute missing values for each variable

#Method Three: Impute with Random Forest
def rfi1(df):
    features_with_missing = df.columns[df.isnull().any()].tolist()
    features_without_missing = df.columns.difference(features_with_missing).tolist()
    for feature in features_with_missing:
        df_copy = df.copy()
        target = df_copy[feature]
        features = df_copy[features_without_missing + [feature]]
        known_features = features[features[feature].notnull()]
        unknown_features = features[features[feature].isnull()]
        # Impute with Random Forest
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(known_features.drop(columns=[feature]), known_features[feature])
        imputed_values = rf.predict(unknown_features.drop(columns=[feature]))
        # Fill the vacancy with imputed data
        df.loc[df[feature].isnull(), feature] = imputed_values
    return df

def rfi(data):
    for column in data.columns:
        # split into  known_data and unknown_data
        known_data = data[data[column].notnull()]
        unknown_data = data[data[column].isnull()]
        features = known_data.drop(column, axis=1)
        target = known_data[column]
        # Impute with HistGradientBoostingRegressor
        imputer = SimpleImputer(strategy='mean')
        imputed_values = imputer.fit_transform(features)
        # Fill the vacancy with imputed data
        data.loc[data[column].isnull(), column] = imputed_values[len(known_data):, features.columns.get_loc(column)]
    return data

imputed_df2=rfi1(data)


# Integrate three imputation methods with Resampling
def SVDi(data,k):
    U, s, VT = np.linalg.svd(data)
    # Set the number of retained singular values, which can be adjusted as needed
    # econstruct the matrix using the top k singular values and their corresponding left singular vectors and right singular vectors
    reconstructed_data = pd.DataFrame(U[:, :k] @ np.diag(s[:k]) @ VT[:k, :])
    # replace missing values in the original DataFrame
    data_filled = data.copy()
    data_filled[data.isna()] = reconstructed_data[data.isna()]
    return data_filled
imputed_df3=SVDi(data,2)
df_avg = imputed_df1.add(imputed_df2).add(imputed_df3).div(3)
print(data.describe())
df_avg=pd.DataFrame(df_avg)
imputed_df1=pd.DataFrame(imputed_df1)
imputed_df2=pd.DataFrame(imputed_df2)
imputed_df3=pd.DataFrame(imputed_df3)
print(imputed_df1.describe())
print(imputed_df2.describe())
print(imputed_df3.describe())
df_dropna = data.dropna()
# replace missing values in the original DataFrame
df_resampled = df_dropna.sample(n=len(data), replace=True)
# merge the resampled samples with the original dataset
df_f = pd.concat([df_resampled, data.loc[data.isnull().any(axis=1)]], ignore_index=True)


#Before calculating the KL divergence, it is necessary to perform a smoothing process
def kl_divergence(p, q, smooth=1e-3):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    # adding a small constant
    p_smooth = p + smooth
    q_smooth = q + smooth
    # normalizing the probability distribution
    p_normalized = p_smooth / np.sum(p_smooth)
    q_normalized = q_smooth / np.sum(q_smooth)
    # computing the KL divergence
    kl = np.sum(p_normalized * np.log(p_normalized / q_normalized))
    kl=abs(kl)
    return kl


from scipy.special import kl_div

def js_divergence(p, q):
    m = (p + q) / 2.0
    kl_pm = kl_div(p, m)
    kl_qm = kl_div(q, m)
    jsd = 0.5 * (np.sum(kl_pm) + np.sum(kl_qm))
    jsd=abs(jsd)
    return jsd

def cross_entropy(p, q):
    assert len(p) == len(q)
    # Convert the probability distribution to a numpy array.
    p = np.array(p)
    q = np.array(q)
    # calculate the cross_entropy
    cross_entropy = -np.sum(p * np.log(q))

    return cross_entropy

df_avg.to_csv("note.csv")
from scipy.stats import chi2_contingency
from scipy.stats import chisquare
def cross_entropy(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return -np.sum(p * np.log(q))

kl_mat = pd.DataFrame(0, index=range(4), columns=range(len(col_nm)))
def kl(data1,i):
    for j in range(len(col_nm)):
        x=data_mean.iloc[:,j]
        y=data1.iloc[:,j]
        #kl_mat.iloc[i, j] = p
        kl_mat.iloc[i,j]=cross_entropy(x,y)

kl(imputed_df1,0)
kl(imputed_df2,1)
kl(imputed_df3,2)
kl(df_avg,3)


#df_avg.to_csv("note.csv")

sns.heatmap(kl_mat, annot=True, cmap='coolwarm')
plt.xticks(range(len(col_nm) + 1), missing_percentages.index, rotation=90)
#plt.yticks(range(len(col_nm) + 1), missing_percentages.index, rotation=0)
plt.xlabel("Missing_Matrix.iloc[:,i]")
plt.show()
kl_mat=pd.DataFrame(kl_mat)
kl_mat.to_csv("note1.csv")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
def kfold(k,model):
    # MAPE for evaluation
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    y = train_target.values.ravel()
    scores = cross_validate(model, train_data, y, cv=k, scoring=scoring)
    # 5 fold cross validation
    accuracy = scores['test_accuracy'].mean()
    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    f1 = scores['test_f1'].mean()
    # mean storage as matrix
    avg_matrix = np.array([k, accuracy, precision, recall, f1])
    print("Average Matrix:", avg_matrix)


#Prediction Model (Decision Tree, MLP, XGBoost, Catboost)
#Decision Tree for prediction
scaler = StandardScaler()
data2 = scaler.fit_transform(train_data)
X_train, X_test, y_train, y_test = train_test_split(data2, train_target, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

X_train=pd.DataFrame(X_train)
#tree_rules = export_text(model, feature_names=X_train.columns.tolist())
#print(tree_rules)
node_count = model.tree_.node_count
print(node_count)



def print_tree(tree, feature_names):
    stack = [(0, -1)]  # original state
    while stack:
        node_id, depth = stack.pop()

        # Retrieve the splitting feature and threshold of a node
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        # determine the node type
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # leaf node
            class_counts = tree.value[node_id].squeeze()
            class_labels = np.arange(len(class_counts))
            class_results = [f"{label}: {count}" for label, count in zip(class_labels, class_counts)]
            indent = "  " * depth
            print(f"{indent}leaf node -> {', '.join(class_results)}")
        else:
            # internal node
            indent = "  " * depth
            print(f"{indent}feature {feature_names[feature_idx]} <= {threshold}")
            stack.append((tree.children_left[node_id], depth + 1))
            print(f"{indent}feature {feature_names[feature_idx]} > {threshold}")
            stack.append((tree.children_right[node_id], depth + 1))


print_tree(model.tree_, X_train.columns.tolist())

# Compute the confusion matrix
y_t = model.predict(X_train)
cm1 = confusion_matrix(y_t, y_train)
cm = confusion_matrix(y_test, y_pred)
print("cm:",cm)
print("cm1:",cm1)
best_feature = model.tree_.feature[0]
print("Optimal splitting feature:", best_feature)
kfold(10,model)


from sklearn.model_selection import train_test_split
scaler = StandardScaler()
data2 = scaler.fit_transform(train_data)
X_train, X_test, y_train, y_test = train_test_split(data2, train_target, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeRegressor
reg_tree = DecisionTreeClassifier()
reg_tree.fit(X_train, y_train)
y_pred = reg_tree.predict(X_test)
y_t=reg_tree.predict(X_train)
print(y_pred)
print(len(y_pred))
cm1 = confusion_matrix(y_pred, y_test)
cm=confusion_matrix(y_t, y_train)
print(cm1)
print(cm)
cm=pd.DataFrame(cm)
cm1=pd.DataFrame(cm1)
cm.to_csv("note.csv")

mse = mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
mae = mean_absolute_error(y_pred, y_test)
print([mse,rmse,mae])
y_pred=pd.DataFrame(y_pred)
y_pred.to_csv("note1.csv")
print("FINISH！")


#MLP
param_grid = {
    'hidden_layer_sizes': [(64,32), (128,64), (256,128)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter':[500],
}



model = MLPClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)
print('Best Parameters:', grid_search.best_params_)
print('Best Score:', -grid_search.best_score_)

# predict with the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
def cm(model1):
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    y_t = model1.predict(X_train)
    cm1 = confusion_matrix(y_t, y_train)
    cm = confusion_matrix(y_test, y_pred)
    print("cm:",cm)
    print("cm1:",cm1)
cm(best_model)

mlp = MLPClassifier(hidden_layer_sizes=(128,),
                   activation='relu',
                   alpha=0.001,
                   solver='adam',
                   learning_rate='constant',
                    max_iter=2000
                   )
kfold(10,mlp)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
def cm(model1):
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    y_t = model1.predict(X_train)
    cm1 = confusion_matrix(y_t, y_train)
    cm = confusion_matrix(y_test, y_pred)
    print("cm:",cm)
    print("cm1:",cm1)
cm(mlp)
kfold(10,mlp)
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

param_grid = {
    'learning_rate': [0.01, 0.1, 1.0],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'lambda': [0.1, 1.0, 10.0],
    'alpha': [0.1, 1.0, 10.0]
}


# XGBoost
xgb = XGBClassifier()
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# print best parameters and scores
print("Best Parameters: ", grid_search.best_params_)
print("Best Score (MSE): ", -grid_search.best_score_)
gbr=GradientBoostingClassifier(
    learning_rate=0.1,         # 学习率，控制每棵树的贡献程度
    n_estimators=100,          # 决策树的数量
    subsample=0.8,             # 训练每棵树时使用的子样本比例
    criterion='friedman_mse',  # 分裂节点的质量评估准则，可选参数：'friedman_mse'（默认）、'mse'或'mae'
    min_samples_split=6,       # 节点分裂所需的最小样本数
    min_samples_leaf=2,        # 叶节点所需的最小样本数
    max_depth=7,               # 决策树的最大深度
    max_features='sqrt',         # 每棵树考虑的最大特征数，可选参数：'auto'（默认）、'sqrt'、'log2'或None
)
def cm(model1):
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    y_t = model1.predict(X_train)
    cm1 = confusion_matrix(y_t, y_train)
    cm = confusion_matrix(y_test, y_pred)
    print("cm:",cm)
    print("cm1:",cm1)
cm(gbr)
kfold(10,gbr)

xgb= XGBClassifier(
    n_estimators=100,  # 决策树数量
    learning_rate=0.1,  # 学习率
    max_depth=7,  # 决策树最大深度
    subsample=1.0,  # 每个决策树样本的比例
    colsample_bytree=0.8,  # 每个决策树特征的比例
    reg_alpha=1.0,  # L1 正则化系数
    reg_lambda=0.1,  # L2 正则化系数
)
cm(xgb)
kfold(10,xgb)


param_grid = {
    'learning_rate': [0.01, 0.1, 1.0],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'lambda': [0.1, 1.0, 10.0],
    'alpha': [0.1, 1.0, 10.0]
}

#practice
xgb = XGBClassifier()
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# print best parameters and scores
print("Best Parameters: ", grid_search.best_params_)
print("Best Score (MSE): ", -grid_search.best_score_)

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# CatBoost
catboost_classifier = CatBoostClassifier()
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.1, 0.01, 0.001],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [0.1, 1, 10]
}
grid_search = GridSearchCV(estimator=catboost_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# print best parameters and scores
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
cbc = CatBoostClassifier(iterations=1000,
                           learning_rate=0.03,
                           depth=6,
                           l2_leaf_reg=3,
                           colsample_bylevel=1,
                           random_seed=42)
cm(cbc)
kfold(10,cbc)
