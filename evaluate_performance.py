import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# 1. Load the model and a test dataset
model = load_model('crypto_predictor_model.h5')
df = pd.read_csv('bitcoin_historic_data.csv') # Or any coin you want to test

# 2. Prepare the data exactly like we did for training
scaler = joblib.load('data_scaler.pkl')
scaled_data = scaler.fit_transform(df[['price', 'MA7', 'MA30']])
X_test = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

# 3. Make Predictions
predictions_scaled = model.predict(X_test)

# 4. Inverse Transform to get real dollar amounts
# We need a dummy scaler for the target to reverse the math
target_scaler = MinMaxScaler()
target_scaler.fit(df[['target_price']])
predictions = target_scaler.inverse_transform(predictions_scaled)

# 5. Plot the Results
plt.figure(figsize=(12, 6))
plt.plot(df['target_price'].values[-50:], label='Actual Price', color='blue')
plt.plot(predictions[-50:], label='Predicted Price', color='red', linestyle='--')
plt.title('AI Crypto Prediction: Actual vs Predicted (Last 50 Days)')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Calculate the error
mae = mean_absolute_error(df['target_price'].values[-50:], predictions[-50:])
print(f"📊 Mean Absolute Error: ${mae:.2f}")

# Optional: Percentage error
mean_price = df['target_price'].values[-50:].mean()
accuracy_pct = (1 - (mae / mean_price)) * 100
print(f"🎯 Model Accuracy: {accuracy_pct:.2f}%")