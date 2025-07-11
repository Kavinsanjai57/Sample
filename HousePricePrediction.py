
# House Price Prediction Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
try:
    df_housing = pd.read_csv('Housing.csv')
    print(df_housing.head())  # Sample output
    print(df_housing.shape)   # (545, 13)
except FileNotFoundError:
    print("Error: 'Housing.csv' not found.")

# Explore Data
print(df_housing.info())
print(df_housing.isnull().sum())
print(df_housing.describe())

# Numerical feature distributions
plt.figure(figsize=(12, 8))
for i, col in enumerate(['price', 'area', 'bedrooms', 'bathrooms']):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df_housing[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Relationships with price
plt.figure(figsize=(12, 8))
for i, col in enumerate(['area', 'bedrooms', 'bathrooms']):
    plt.subplot(1, 3, i + 1)
    plt.scatter(df_housing[col], df_housing['price'])
    plt.xlabel(col)
    plt.ylabel('price')
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.show()

# Outlier removal using IQR
numerical_features = ['price', 'area']
for feature in numerical_features:
    Q1 = df_housing[feature].quantile(0.25)
    Q3 = df_housing[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_housing = df_housing[(df_housing[feature] >= lower) & (df_housing[feature] <= upper)]

# Categorical encoding
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
for feature in categorical_features:
    dummies = pd.get_dummies(df_housing[feature], prefix=feature, drop_first=True)
    df_housing = pd.concat([df_housing, dummies], axis=1)
    df_housing.drop(feature, axis=1, inplace=True)

# Furnishing status encoding
df_housing['furnishingstatus'] = df_housing['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

# Feature Engineering
df_housing['area_bedrooms'] = df_housing['area'] * df_housing['bedrooms']
df_housing['area_bathrooms'] = df_housing['area'] * df_housing['bathrooms']
df_housing['bedrooms_bathrooms'] = df_housing['bedrooms'] * df_housing['bathrooms']
df_housing['area_sq'] = df_housing['area'] ** 2
df_housing['area_cubed'] = df_housing['area'] ** 3
df_housing['log_price'] = np.log1p(df_housing['price'])
df_housing['log_area'] = np.log1p(df_housing['area'])

# Split Data
X = df_housing.drop('log_price', axis=1)
y = df_housing['log_price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)

# Train Models
model_linear = LinearRegression()
model_tree = DecisionTreeRegressor(random_state=42)
model_forest = RandomForestRegressor(random_state=42)
model_gbm = GradientBoostingRegressor(random_state=42)

model_linear.fit(X_train, y_train)
model_tree.fit(X_train, y_train)
model_forest.fit(X_train, y_train)
model_gbm.fit(X_train, y_train)

# Hyperparameter Tuning
param_grid_tree = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_forest = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

param_grid_gbm = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1]
}

grid_tree = GridSearchCV(model_tree, param_grid_tree, cv=5, scoring='neg_mean_squared_error')
grid_tree.fit(X_train, y_train)

grid_forest = RandomizedSearchCV(model_forest, param_grid_forest, n_iter=10, cv=5, scoring='neg_mean_squared_error')
grid_forest.fit(X_train, y_train)

grid_gbm = RandomizedSearchCV(model_gbm, param_grid_gbm, n_iter=10, cv=5, scoring='neg_mean_squared_error')
grid_gbm.fit(X_train, y_train)

best_models = {
    'tree': grid_tree.best_estimator_,
    'forest': grid_forest.best_estimator_,
    'gbm': grid_gbm.best_estimator_
}

# Evaluate on test set
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> MAE: {mae}, RMSE: {rmse}, R2: {r2}")

# Example output: Best model by RMSE: GBM
