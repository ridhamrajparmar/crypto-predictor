import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout

# 1. Load and Merge Data
all_features = []
all_targets = []

print("Loading datasets...")
for file in os.listdir():
    if file.endswith("_historic_data.csv"):
        df = pd.read_csv(file)
        # Just collect the raw data first
        all_features.append(df[['price', 'MA7', 'MA30']].values)
        all_targets.append(df[['target_price']].values)

# Combine all raw data into one big array
X_raw = np.vstack(all_features)
y_raw = np.vstack(all_targets)

# Now, fit and transform using one global scaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

# We use a separate scaler or a separate fit for the target 
# so we can inverse-transform the price easily later
target_scaler = MinMaxScaler()
y = target_scaler.fit_transform(y_raw)

# Save the fitted scalers
joblib.dump(scaler, 'data_scaler.pkl')
joblib.dump(target_scaler, 'target_scaler.pkl')
print("Scalers fitted and saved!")

# 2. Reshape for LSTM (Samples, Time Steps, Features)
# For now, we use 1 time step per sample for simplicity
X = X.reshape((X.shape[0], 1, X.shape[1]))

# 3. Define the AI Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1) # The predicted price
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train the Model
print("Starting training...")
model.fit(X, y, epochs=20, batch_size=64, validation_split=0.1)

# 5. Save the Model
model.save('crypto_predictor_model.h5')
print("Model saved as crypto_predictor_model.h5")

joblib.dump(scaler, 'data_scaler.pkl')
print("Scaler saved as data_scaler.pkl!")