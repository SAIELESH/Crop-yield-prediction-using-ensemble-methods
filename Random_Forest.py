import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Data Loading
df = pd.read_csv('/content/corn_data_final_no_county.csv')

df = df.drop(['loc_ID', 'year'], axis=1)

# Separating the features and target
X = df.drop('yield', axis=1)
y = df['yield']

# Dataset Spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Defining parameter space for the Bayesian search
param_space = {
    'n_estimators': (10, 100),
    'max_depth': (1, 50),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

# Model Definition
model = RandomForestRegressor(random_state=42)

# Creating the BayesSearchCV object
opt = BayesSearchCV(
    model,
    param_space,
    n_iter=50,  # Increase iterations for larger parameter space
    cv=KFold(n_splits=10),
    n_jobs=-1,
    random_state=42,
)

# Performing the Bayesian search
opt.fit(X_train, y_train)

print("Best Parameters: ", opt.best_params_)

# Using the best model to make predictions
best_model = opt.best_estimator_
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

# Calculate RRMSE
train_rrmse = train_rmse / np.mean(y_train)
test_rrmse = test_rmse / np.mean(y_test)

# Print the evaluation metrics
print(f'Training RMSE: {train_rmse}, R2: {train_r2}, RRMSE: {train_rrmse}')
print(f'Testing RMSE: {test_rmse}, R2: {test_r2}, RRMSE: {test_rrmse}')

# Plot the error distribution for training data
plt.figure(figsize=(10,6))
plt.hist(y_train - train_preds, bins=20, edgecolor='black')
plt.title('Error Distribution: Training Data')
plt.xlabel('Prediction Error')
plt.show()

# Plot the error distribution for testing data
plt.figure(figsize=(10,6))
plt.hist(y_test - test_preds, bins=20, edgecolor='black')
plt.title('Error Distribution: Testing Data')
plt.xlabel('Prediction Error')
plt.show()

# Scatter plot of actual vs predicted for training data
plt.figure(figsize=(10,6))
plt.scatter(y_train, train_preds)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted: Training Data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
plt.show()

# Scatter plot of actual vs predicted for testing data
plt.figure(figsize=(10,6))
plt.scatter(y_test, test_preds)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted: Testing Data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()