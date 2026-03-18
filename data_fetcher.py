import requests
import pandas as pd

def fetch_crypto_data(coin_id):
    # Update to the 'pro-api' subdomain which handles the Demo keys
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': '365',
        'interval': 'daily'
    }
    
    # Add your API Key to the headers
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-zCXgaULdeMr7zcDx9CfUrqgH" # Paste your key here
    }
    
    print(f"Fetching data for {coin_id}...")
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    bitcoin_data = fetch_crypto_data('bitcoin')
    if bitcoin_data is not None:
        print(bitcoin_data.tail()) # tail() shows the most recent days