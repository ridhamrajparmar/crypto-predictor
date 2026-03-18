from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import io
import requests

# 1. Initialize the Flask app
app = Flask(__name__)

# 2. Load your AI assets (Make sure these files are in the same folder!)
model = load_model('crypto_predictor_model.h5')
scaler = joblib.load('data_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Get your local coin list from CSVs
    raw_coins = [f.replace('_historic_data.csv', '').upper() for f in os.listdir() if f.endswith('_historic_data.csv')]
    
    # Optional: Fetch high-quality icons from CoinGecko to pass to the frontend
    # For now, we'll keep the list simple and handle the logic in the HTML for speed.
    return render_template('dashboard.html', coins=raw_coins)

@app.route('/predict', methods=['POST'])
def predict():
    coin_name = request.json['coin'].lower()
    file_path = f"{coin_name}_historic_data.csv"
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Data not found'})

    df = pd.read_csv(file_path)
    
    # Get last 30 days of prices for the chart
    chart_data = df['price'].tail(30).tolist()
    
    # AI Prediction Logic
    latest_data = df[['price', 'MA7', 'MA30']].tail(1).values
    scaled_input = scaler.transform(latest_data)
    X_input = scaled_input.reshape((1, 1, 3))
    prediction_scaled = model.predict(X_input)
    prediction_final = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
    
    return jsonify({
        'current_price': df['price'].iloc[-1],
        'predicted_price': float(prediction_final),
        'chart_values': chart_data
    })

# 3. New Route for the "Little Graph"
@app.route('/plot/<coin>.png')
def plot_png(coin):
    file_path = f"{coin.lower()}_historic_data.csv"
    if not os.path.exists(file_path):
        return "File not found", 404
        
    df = pd.read_csv(file_path)
    recent_data = df.tail(30) # Shows the last 30 days

    plt.figure(figsize=(10, 4), facecolor='white')
    plt.plot(recent_data['price'].values, color='#7c3aed', linewidth=3)
    plt.fill_between(range(len(recent_data)), recent_data['price'], color='#7c3aed', alpha=0.1)
    plt.axis('off') 
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)