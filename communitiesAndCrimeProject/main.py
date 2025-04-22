import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor

#I selected the 127 col (violentCrimeRate)
df = pd.read_csv("communities+and+crime/communities.data", header=None, na_values="?")
print(df.head())
print(df.isnull().sum())
print(df.isnull().sum().sum())

#Decision Tree
df_filled = df.fillna(df.median(numeric_only=True))
print(df_filled.head())
print(df_filled[0])

df_num = df_filled.select_dtypes(include=[np.number])

X = df_num.iloc[:, :-1]
y = df_num.iloc[:, -1] #violentCrimeRate

#Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train/test splits
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Decision Tree ---
tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)

print(f"MSE: {tree_mse}")
print(f"R^2 Score: {tree_r2}")

plt.figure(figsize=(20, 10))
plot_tree(tree_model, filled=True, feature_names=X.columns, rounded=True)
plt.title("Decision Tree for Crime Rate Prediction")
plt.show()

# --- KNN Regressor ---
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_pred)
knn_r2 = r2_score(y_test, knn_pred)

# --- Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# --- Plot MSE and R^2 Comparison
models = ['Decision Tree', 'KNN', 'Random Forest']
mses = [tree_mse, knn_mse, rf_mse]
r2s = [tree_r2, knn_r2, rf_r2]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

#MSE bar chart
ax[0].bar(models, mses, color="salmon")
ax[0].set_title('MSE Comparison (Regression)')
ax[0].set_ylabel('Mean Squared Error')

#R^2 bar chart
ax[1].bar(models, r2s, color='skyblue')
ax[1].set_title('R^2 Score Comparison (Regression)')
ax[1].set_ylabel('R^2 Score')

plt.tight_layout()
plt.show()

# --- Classification Models ---
# First -> convert y to binary -> 1 = high crime, 0 = low crime
median_crime = y.median()
y_binary = (y > median_crime).astype(int)
y_train_bin, y_test_bin = train_test_split(y_binary, test_size=0.2, random_state=42)

# --- Logistic Regression ---
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train_bin)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test_bin, log_pred)
log_cm = confusion_matrix(y_test_bin, log_pred)

# --- Naive Bayes ---
nb_model = GaussianNB()
nb_model.fit(X_train, y_train_bin)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test_bin, nb_pred)
nb_cm = confusion_matrix(y_test_bin, nb_pred)

# --- Plot Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Logistic Regression\nAccuracy: {log_acc:.2f}')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title(f'Naive Bayes\nAccuracy: {nb_acc:.2f}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
