import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("communities+and+crime/communities.data", header=None, na_values="?")
print(df.head())
print(df.isnull().sum())
print(df.isnull().sum().sum())

df_filled = df.fillna(df.median(numeric_only=True))
print(df_filled.head())
print(df_filled[0])

df_num = df_filled.select_dtypes(include=[np.number])

X = df_num.iloc[:, :-1]
y = df_num.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")

plt.figure(figsize=(20, 10))
plot_tree(tree_model, filled=True, feature_names=X.columns, rounded=True)
plt.title("Decision Tree for Crime Rate Prediction")
plt.show()