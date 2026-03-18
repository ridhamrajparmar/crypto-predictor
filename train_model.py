import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout

# 1. Load your dataset (starting with one coin for testing)
df = pd.read_csv('bitcoin_historic_data.csv')

# 2. Select Features (MA7, MA30, and current Price)
features = df[['price', 'MA7', 'MA30']].values
target = df['target_price'].values.reshape(-1, 1)

# 3. Scale the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_features = scaler_x.fit_transform(features)
scaled_target = scaler_y.fit_transform(target)

# 4. Split into Training and Testing sets
# Note: For time-series, shuffle=False is vital to keep the chronological order
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, scaled_target, test_size=0.2, shuffle=False
)

# 5. Reshape for LSTM [samples, time_steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 6. Build the Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Train!
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Replace 'scaler' with whatever you named your MinMaxScaler variable
scaler = MinMaxScaler()
joblib.dump(scaler, 'data_scaler.pkl')
print("Scaler saved as data_scaler.pkl!")