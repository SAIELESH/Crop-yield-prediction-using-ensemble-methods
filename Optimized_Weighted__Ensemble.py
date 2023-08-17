import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score

# Data loading
df = pd.read_csv('/content/corn_data_final_no_county.csv')

# Separating out the target variable, features, and track columns
y = df['yield']
X = df.drop(columns=['yield', 'loc_ID', 'year'])
track_cols = df[['loc_ID', 'year']]

# Defining base learners with different parameters
base_learners = [RandomForestRegressor(n_estimators=50, max_depth=10, max_features='sqrt', min_samples_leaf=7) for _ in range(10)]

# Spliting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# K-Fold cross-validation
kf = KFold(n_splits=10)

# Out-of-bag predictions Collection
oob_predictions = []

for train_index, test_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    fold_preds = np.zeros((len(base_learners), len(X_val_fold)))
    for i, model in enumerate(base_learners):
        model.fit(X_train_fold, y_train_fold)
        fold_preds[i, :] = model.predict(X_val_fold)

    oob_predictions.append(fold_preds)

# Concatenating the predictions 
oob_predictions = np.concatenate(oob_predictions, axis=1)

# Defining the objective function for the optimization 

def objective_function(W, y_true, y_preds):
    return ((y_true - np.dot(W, y_preds)) ** 2).mean()

# Defining constraints and bounds
cons = ({'type': 'eq', 'fun': lambda W: 1 - np.sum(W)})
bounds = [(0, 1) for _ in range(len(base_learners))]

# Performing the optimization
result = minimize(objective_function,
                  x0=np.array([1/len(base_learners)] * len(base_learners)),
                  args=(y_train, oob_predictions),
                  method='SLSQP',
                  bounds=bounds,
                  constraints=cons)


optimal_weights = result.x

# Fitting the base learners on the full training set and predicting on the test set

train_preds = np.zeros((len(base_learners), len(X_train)))
test_preds = np.zeros((len(base_learners), len(X_test)))
for i, model in enumerate(base_learners):
    model.fit(X_train, y_train)
    train_preds[i, :] = model.predict(X_train)
    test_preds[i, :] = model.predict(X_test)

# Computing the ensemble predictions
ensemble_prediction_train = np.dot(optimal_weights, train_preds)
ensemble_prediction_test = np.dot(optimal_weights, test_preds)


rmse_train = np.sqrt(mean_squared_error(y_train, ensemble_prediction_train))
rmse_test = np.sqrt(mean_squared_error(y_test, ensemble_prediction_test))


r2_train = r2_score(y_train, ensemble_prediction_train)
r2_test = r2_score(y_test, ensemble_prediction_test)


rrmse_train = rmse_train / np.mean(y_train)
rrmse_test = rmse_test / np.mean(y_test)


print(f'Train RMSE: {rmse_train}')
print(f'Test RMSE: {rmse_test}')
print()
print()
print(f'Train R2: {r2_train}')
print(f'Test R2: {r2_test}')
print()
print()
print(f'Train RRMSE: {rrmse_train}')
print(f'Test RRMSE: {rrmse_test}')

