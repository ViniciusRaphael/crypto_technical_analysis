import json
from datetime import datetime
from pathlib import Path

import requests as re
import yfinance as yf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_kucoin_symbols():
    resp = re.get('https://api.kucoin.com/api/v2/symbols')
    ticker_list = json.loads(resp.content)
    symbols_list = [ticker['symbol'] for ticker in ticker_list['data'] if str(ticker['symbol'][-4:]) == 'USDT']
    return symbols_list


def fetch_crypto_data(symbols, start_date):
    df_list = []

    for count, symbol in enumerate(symbols):
        print(f'Processing {symbol} ({count + 1} of {len(symbols)})')

        # Get historical data using yfinance for the given symbol since start_date
        crypto_data = yf.Ticker(symbol).history(start=start_date)

        # Add a 'symbol' column to the DataFrame to identify the cryptocurrency
        crypto_data['Symbol'] = symbol

        # Append the DataFrame to the list if it's not empty
        if not crypto_data.empty:
            df_list.append(crypto_data)

    # Concatenate the list of DataFrames into a single DataFrame
    if df_list:
        return pd.concat(df_list)
    else:
        return None


def save_parquet_file(dataframe, file_path):
    table = pa.Table.from_pandas(dataframe)
    pq.write_table(table=table, where=file_path, compression='snappy')


def main():
    # Get a list of cryptocurrency symbols from Kucoin
    kucoin_symbols = get_kucoin_symbols()

    # Remove the last character from each symbol in the list
    # USDT to USD - name convention on yfinance
    crypto_symbols = [symbol[:-1] for symbol in kucoin_symbols]

    # Define start date for historical data retrieval
    start_date = '2018-01-01'

    # Fetch historical cryptocurrency data
    crypto_dataframe = fetch_crypto_data(crypto_symbols, start_date)

    if crypto_dataframe is not None:
        # Specify the folder and file name for saving the Parquet file
        output_folder = 'files'
        output_file = 'crypto_historical_data.parquet'
        output_path = Path(output_folder) / output_file

        # Create the output folder if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame as a Parquet file
        save_parquet_file(crypto_dataframe, output_path)

        print(f"Parquet file saved to {output_path}")
    else:
        print("No data fetched.")

if __name__ == "__main__":
    main()
