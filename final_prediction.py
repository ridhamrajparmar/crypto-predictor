import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

# 1. Load the Brain and the Translator
model = load_model('crypto_predictor_model.h5')
scaler = joblib.load('data_scaler.pkl')

# 2. Load the latest data (e.g., Bitcoin)
df = pd.read_csv('bitcoin_historic_data.csv')

# 3. Get the very last row of data to predict the future
# We take the last available values for price, MA7, and MA30
latest_data = df[['price', 'MA7', 'MA30']].tail(1).values

# 4. Scale it so the AI understands
scaled_input = scaler.transform(latest_data)
X_input = scaled_input.reshape((1, 1, 3))

# ... (after making the prediction) ...

# 5. Make the Prediction
prediction_scaled = model.predict(X_input)

# 6. Translate the answer back to Dollars
# Fix: Extract the single value from the array using .flatten()[0]
target_scaler = joblib.load('target_scaler.pkl') # Load the target-specific scaler
prediction_final = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]

print("-" * 30)
print(f"🚀 LATEST ACTUAL PRICE: ${df['price'].iloc[-1]:,.2f}")
print(f"🔮 AI PREDICTION (7 DAYS LATER): ${prediction_final:,.2f}")
print("-" * 30)