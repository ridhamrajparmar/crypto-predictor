import requests
import pandas as pd
import time
import os  # Necessary for checking if files exist

def get_and_save_data(coin_id, api_key):
    filename = f"{coin_id}_historic_data.csv"
    
    # 1. Check if the dataset already exists locally
    if os.path.exists(filename):
        print(f"⏩ {coin_id} already exists. Skipping...")
        return "skipped"

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': '365', 'interval': 'daily'}
    headers = {"accept": "application/json", "x-cg-demo-api-key": api_key}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            raw_data = response.json()
            df = pd.DataFrame(raw_data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Feature Engineering
            df['MA7'] = df['price'].rolling(window=7).mean()
            df['MA30'] = df['price'].rolling(window=30).mean()
            df['target_price'] = df['price'].shift(-7)
            
            df = df.dropna()
            
            df.to_csv(filename, index=False)
            return "success"
        else:
            print(f"❌ Error for {coin_id}: {response.status_code}")
            return "failed"
            
    except Exception as e:
        print(f"⚠️ Network error for {coin_id}: {e}")
        return "failed"

if __name__ == "__main__":
    my_key = "CG-zCXgaULdeMr7zcDx9CfUrqgH"
    
    # Updated IDs for Polygon (matic-network), Toncoin (the-open-network), and Stacks (blockstack)
    top_30_coins = [
        'bitcoin', 'ethereum', 'tether', 'solana', 'binancecoin', 
        'ripple', 'usd-coin', 'staked-ether', 'cardano', 'avalanche-2', 
        'dogecoin', 'shiba-inu', 'polkadot', 'tron', 'chainlink', 
        'matic-network', 'the-open-network', 'internet-computer', 'bitcoin-cash', 'dai', 
        'uniswap', 'litecoin', 'near', 'aptos', 'blockstack', 
        'ethereum-classic', 'filecoin', 'optimism', 'render-token', 'arbitrum'
    ]
    
    for coin in top_30_coins:
        result = get_and_save_data(coin, my_key)
        
        if result == "success":
            print(f"✅ Successfully saved {coin}")
            # Only wait if we actually made a request to the API
            print("Waiting 30 seconds to respect rate limits...")
            time.sleep(30)
        elif result == "failed":
            # If it failed, wait a few seconds before the next one
            time.sleep(5)
        elif result == "skipped":
            # No API call was made, so no need to wait
            continue