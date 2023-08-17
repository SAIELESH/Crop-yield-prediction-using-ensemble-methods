import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
import seaborn as sns

# Data loading
df = pd.read_csv("/content/corn_data_final_no_county.csv")

# Preprocessing
df = df.drop(['loc_ID', 'year'], axis=1)

# Data Splitting
X = df.drop('yield', axis=1)
y = df['yield']

# Creating a StandardScaler object for the target
scaler_y = StandardScaler()
y = pd.DataFrame(scaler_y.fit_transform(y.values.reshape(-1, 1)), columns=[y.name])

# Applying feature scaling to the features
scaler_X = StandardScaler()
X = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)

# Splitting the data into base and meta datasets
X_base, X_meta, y_base, y_meta = train_test_split(X, y, test_size=0.3, random_state=42)

# Converting y to a 1-D array
y_base = y_base.values.ravel()
y_meta = y_meta.values.ravel()

# Splitting the meta dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)

# Defining base models and extended hyperparameters
base_models = [
    (LGBMRegressor(), {'learning_rate': (0.01, 0.5), 'n_estimators': (100, 500)}),
    (HistGradientBoostingRegressor(), {'learning_rate': (0.01, 0.5), 'max_iter': (100, 500)})
]

# Training base models and get predictions for the meta model
meta_features = pd.DataFrame()

models = []  # Storing best models for later prediction

for i, (model, params) in enumerate(base_models):
    opt = BayesSearchCV(model, params, cv=10, n_iter=50, random_state=0) 
    opt.fit(X_base, y_base)
    best_model = opt.best_estimator_
    models.append(best_model)  # Save the best model
    meta_features[f'model_{i}'] = best_model.predict(X_meta)

# Meta model Training
meta_model = LinearRegression()
meta_model.fit(meta_features, y_meta)

# Preparing test features
test_features = pd.DataFrame()
for i, model in enumerate(models):  # Use saved best models
    test_features[f'model_{i}'] = model.predict(X_test)

# Making predictions
y_train_pred = meta_model.predict(meta_features)
y_test_pred = meta_model.predict(test_features)

# Inverse transform the predictions and the actual 'yield' values
y_train_true = scaler_y.inverse_transform(y_meta.reshape(-1,1))
y_train_pred = scaler_y.inverse_transform(y_train_pred.reshape(-1,1))
y_test_true = scaler_y.inverse_transform(y_test.reshape(-1,1))
y_test_pred = scaler_y.inverse_transform(y_test_pred.reshape(-1,1))


rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
r2_train = r2_score(y_train_true, y_train_pred)
rrmse_train = rmse_train / np.mean(y_train_true)

rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
r2_test = r2_score(y_test_true, y_test_pred)
rrmse_test = rmse_test / np.mean(y_test_true)

print(f"Training RMSE: {rmse_train}, R2: {r2_train}, RRMSE: {rrmse_train}")
print(f"Testing RMSE: {rmse_test}, R2: {r2_test}, RRMSE: {rrmse_test}")

# Compute residuals
train_residuals = y_train_true.flatten() - y_train_pred.flatten()
test_residuals = y_test_true.flatten() - y_test_pred.flatten()

# Set up plots
sns.set(style="whitegrid")

# Plot error distribution for the training data
sns.displot(train_residuals, bins=20, kde=True, color='b')
plt.title('Error Distribution - Training Data')
plt.xlabel('Prediction Error')
plt.show()

# Plot error distribution for the testing data
sns.displot(test_residuals, bins=20, kde=True, color='r')
plt.title('Error Distribution - Testing Data')
plt.xlabel('Prediction Error')
plt.show()

# Scatter plot of Actual vs Predicted for training and testing data
plt.figure(figsize=(10, 8))
plt.scatter(y_train_true, y_train_pred, c='blue', label='Train', alpha=0.3)
plt.scatter(y_test_true, y_test_pred, c='red', label='Test', alpha=0.3)
plt.plot([min(y_train_true), max(y_train_true)], [min(y_train_true), max(y_train_true)], color='black')  # Add regression line
plt.title('Predictions vs Actual - Training & Testing Data')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

plt.show()