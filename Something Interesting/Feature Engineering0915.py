import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取Excel文件
df = pd.read_excel("C:/Users/WeiTh/Downloads/m6.xlsx")
new_data = df['W1'].iloc[10:].reset_index(drop=True)

new_df = pd.DataFrame()
new_df['Current_W'] = new_data
for i in range(len(new_data)):
    current_index = i + 10
    previous_10_rows = df.iloc[current_index-10:current_index]
    flattened_data = previous_10_rows[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7']].values.flatten()
    for j in range(len(flattened_data)):
        new_df.loc[i, f'Prev_{j//7+1}_W{j%7+1}'] = flattened_data[j]

#new_df.to_excel('C:/Users/WeiTh/Downloads/output93.xlsx', index=False)
#print("数据处理完成，已保存到 output_data.xlsx")
#data = pd.read_excel("C:/Users/WeiTh/Downloads/output93.xlsx")

X = data.drop("Current_W", axis=1)
y = data["Current_W"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[-1:])

selector = SelectKBest(score_func=f_regression, k=2)
X_selected = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()]
feature_scores = selector.scores_
feature_df = pd.DataFrame({'Feature': selected_features, 'Score': feature_scores[selector.get_support()]})
feature_df = feature_df.sort_values(by='Score', ascending=False)
print(feature_df)


X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


new_data = np.array([8, 16, 18, 28, 29, 32, 20, 1, 13, 25, 27, 31, 34, 33, 4, 10, 
                     15, 20, 32, 38, 26, 28, 32, 34, 37, 45, 49, 7, 6, 8, 10, 23, 32, 
                     45, 3, 17, 22, 26, 36, 39, 45, 14, 2, 3, 16, 25, 38, 48, 6, 5, 36, 
                     39, 44, 48, 49, 32, 6, 18, 31, 33, 44, 49, 2, 5, 18, 23, 24, 29, 49, 11]).reshape(1, -1)  # Reshape为(1, 3)的二维数组
new_data = X[-5:-1]
new_data_scaled = scaler.transform(new_data)

new_data_selected = selector.transform(new_data_scaled)

prediction = mlp.predict(new_data_selected)

print("MLP预测结果:", prediction)

"----------------------"

from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

df = pd.read_excel("C:/Users/WeiTh/Downloads/m6.xlsx")
y_df = df.iloc[10:].reset_index(drop=True)
print(y_df.head(5))

data = new_df
X = data.drop("Current_W", axis=1)
y = y_df["W1"]
selector = SelectKBest(score_func=f_regression, k=3)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
feature_scores = selector.scores_
feature_df = pd.DataFrame({'Feature': selected_features, 'Score': feature_scores[selector.get_support()]})
feature_df = feature_df.sort_values(by='Score', ascending=False)
print(feature_df)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)
predictions = reg_tree.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (Decision Tree):", mse)

new_data = np.array([8, 16, 18, 28, 29, 32, 20, 1, 13, 25, 27, 31, 34, 33, 4, 10, 
                     15, 20, 32, 38, 26, 28, 32, 34, 37, 45, 49, 7, 6, 8, 10, 23, 32, 
                     45, 3, 17, 22, 26, 36, 39, 45, 14, 2, 3, 16, 25, 38, 48, 6, 5, 36, 
                     39, 44, 48, 49, 32, 6, 18, 31, 33, 44, 49, 2, 5, 18, 23, 24, 29, 49, 11])

new_data = np.array(new_data[-5:-1]).reshape(1, -1)  
new_data_selected = selector.transform(new_data)
prediction = reg_tree.predict(new_data_selected)

print("回归树预测结果:", prediction)
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


'------------------------------------'
# 读取Excel文件
df = pd.read_excel("C:/Users/WeiTh/Downloads/Mark_Six.xlsx")
new_data = df['W1'].iloc[-1500:].reset_index(drop=True)
df2 = df.iloc[-1500:].reset_index(drop=True)
new_df = pd.DataFrame()
new_df['Current_W'] = new_data
for i in range(len(new_data)):
    current_index = i + 10
    previous_rows = df2.iloc[current_index-10:current_index]
    flattened_data = previous_rows[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7']].values.flatten()
    for j in range(len(flattened_data)):
        new_df.loc[i, f'Prev_{j//7+1}_W{j%7+1}'] = flattened_data[j]

new_df.to_excel('C:/Users/WeiTh/Downloads/output94.xlsx', index=False)
print("数据处理完成，已保存到 output_data.xlsx")
