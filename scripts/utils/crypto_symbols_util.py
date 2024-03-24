import json
import requests as re

def get_kucoin_symbols():
    resp = re.get('https://api.kucoin.com/api/v2/symbols')
    ticker_list = json.loads(resp.content)
    symbols_list = [ticker['symbol'] for ticker in ticker_list['data'] if str(ticker['symbol'][-4:]) == 'USDT']

    return symbols_list