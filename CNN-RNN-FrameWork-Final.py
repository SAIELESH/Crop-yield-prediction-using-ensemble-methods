import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, AveragePooling1D, Flatten, Input, concatenate, Dropout, BatchNormalization, LSTM
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Data loading
df = pd.read_csv("/content/corn_data_final_no_county.csv")


df.drop(['loc_ID', 'year'], axis=1, inplace=True)

# Creating Transformer
weather_components = [f'W_{i}_{j}' for i in range(1, 7) for j in range(1, 53)]
soil_components = [
    column_name for column_name in df.columns if column_name.startswith('bdod') or column_name.startswith('cec') or
    column_name.startswith('cfvo') or column_name.startswith('clay') or column_name.startswith('nitrogen') or
    column_name.startswith('ocd') or column_name.startswith('ocs') or column_name.startswith('phh2o') or
    column_name.startswith('sand') or column_name.startswith('silt') or column_name.startswith('soc')
] 
plant_time = [
    'P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10', 'P_11', 'P_12', 'P_13', 'P_14'
] 

# Defining preprocessor
preprocessor = make_column_transformer(
    (MinMaxScaler(), weather_components),
    (MinMaxScaler(), soil_components),
    (OneHotEncoder(handle_unknown="ignore"), plant_time)
)

# Preprocessing the data
X = df[weather_components + soil_components + plant_time]
y = df['yield']
X = preprocessor.fit_transform(X)

# Reshape to match CNN inputs
X_weather = X[:, :312].reshape(-1, 52, 6)
X_soil = X[:, 312:312 + len(soil_components)].reshape(-1, len(soil_components), 1)
X_time = X[:, 312 + len(soil_components):]

# Data Splitting
X_weather_train, X_weather_test, X_soil_train, X_soil_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
    X_weather, X_soil, X_time, y, test_size=0.2, random_state=1
)

# Defining the model

input_weather = Input(shape=(52, 6))
input_soil = Input(shape=(len(soil_components), 1))
input_time = Input(shape=(X_time.shape[1],))

# Weather branch
w = LSTM(64, return_sequences=True)(input_weather)

# Soil branch
s = LSTM(64)(input_soil)

# Plant time branch
p = Dense(64, activation='relu')(input_time)

# Concatenate branches
concat = concatenate([w[:, -1, :], s, p])

# Dense layers
dense = Dense(100, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(concat)
dense = Dropout(0.3)(dense)
dense = Dense(75, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(dense)
dense = Dropout(0.3)(dense)

# Output layer
output = Dense(1)(dense)

# Creating the model
model = Model(inputs=[input_weather, input_soil, input_time], outputs=output)

# Model Compilation
model.compile(optimizer=RMSprop(learning_rate=0.0003), loss='mse')

# Defining early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Training the model with early stopping and learning rate scheduler callback
history = model.fit([X_weather_train, X_soil_train, X_time_train], y_train, epochs=35000, batch_size=25, verbose=2, validation_split=0.2,
          callbacks=[early_stopping])


# Model Evaluation
train_preds = model.predict([X_weather_train, X_soil_train, X_time_train]).flatten()
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_r2 = r2_score(y_train, train_preds)
train_rrmse = train_rmse / np.mean(y_train) 

test_preds = model.predict([X_weather_test, X_soil_test, X_time_test]).flatten()
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_r2 = r2_score(y_test, test_preds)
test_rrmse = test_rmse / np.mean(y_test) 

print('Train RMSE: %.3f, Test RMSE: %.3f' % (train_rmse, test_rmse))
print('Train R2: %.3f, Test R2: %.3f' % (train_r2, test_r2))
print('Train RRMSE: %.3f, Test RRMSE: %.3f' % (train_rrmse, test_rrmse))

