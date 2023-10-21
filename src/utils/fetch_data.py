import yfinance as yf

def fetch_data():
    btc_data = yf.download('BTC-USD', start='2015-01-01', end='2023-01-01')
    btc_data.to_csv('./data/btc_data.csv')

fetch_data()