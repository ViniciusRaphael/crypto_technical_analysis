from pathlib import Path
import pandas as pd
import numpy as np

import warnings

import re as reg
import json
import requests as re
import yfinance as yf

warnings.filterwarnings("ignore")


class DataIngestion():

    def __init__(self) -> None:
        pass

        
    def get_kucoin_symbols(self):
        """
        This function fetches all trading symbols from the KuCoin API and filters out those symbols 
        that have 'USDT' as their trading pair. It returns a list of these symbols.

        Returns:
        list of str: A list of trading symbols that have 'USDT' as their trading pair.
        """
        # Send a GET request to the KuCoin API to fetch all trading symbols
        resp = re.get('https://api.kucoin.com/api/v2/symbols')
        
        # Parse the response content to a JSON object
        ticker_list = json.loads(resp.content)
        
        # Filter and collect symbols that end with 'USDT'
        symbols_list = [ticker['symbol'] for ticker in ticker_list['data'] if str(ticker['symbol'][-4:]) == 'USDT']
        
        return symbols_list


    def fetch_crypto_data(self, symbols, start_date):
        """
        This function fetches historical cryptocurrency data for a list of symbols starting from a specified date.
        It uses the yfinance library to retrieve the data and returns a concatenated DataFrame containing the data 
        for all the specified symbols.

        Parameters:
        symbols (list of str): List of cryptocurrency symbols to fetch data for.
        start_date (str): The start date from which to fetch historical data (in 'YYYY-MM-DD' format).

        Returns:
        pd.DataFrame: A DataFrame containing historical data for all specified symbols, or None if no data is fetched.
        """
        df_list = []

        for count, symbol in enumerate(symbols):
            print(f'Processing {symbol} ({count + 1} of {len(symbols)})')
            try:
                # Get historical data using yfinance for the given symbol since start_date
                crypto_data = yf.Ticker(symbol).history(start=start_date)
                # Add a 'Symbol' column to the DataFrame to identify the cryptocurrency
                crypto_data['Symbol'] = symbol

                # Append the DataFrame to the list if it's not empty
                if not crypto_data.empty:
                    df_list.append(crypto_data)
            except Exception as e:
                # Handle exceptions, if any, and continue with the next symbol
                print(f'Error processing {symbol}: {e}')
                pass

        # Concatenate the list of DataFrames into a single DataFrame if the list is not empty
        if df_list:
            return pd.concat(df_list)
        else:
            return None

    def correcting_numbers_discrepancy(self, dataframe):
        """
        This function corrects discrepancies in numerical columns ('Open', 'High', 'Low', 'Close', 'Volume', 'Dividends')
        by replacing the current value with the previous value within each 'Symbol' group if the discrepancy ratio is 
        greater than or equal to 5(500%). 
        
        Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing stock market data with columns 'Symbol', 'Open', 'High', 
                                'Low', 'Close', 'Volume', and 'Dividends'.
        
        Returns:
        pd.DataFrame: The corrected DataFrame with discrepancies addressed.
        """
        columns_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']
        
        for column in columns_list:
            # Create a new column with the previous value of the current column within each 'Symbol' group
            dataframe[f'new_{column}'] = dataframe.groupby('Symbol')[f'{column}'].transform(lambda x: x.shift(1))
            
            # Calculate the corrected value if the discrepancy ratio is greater than or equal to 5
            dataframe[f'{column}_2'] = np.where((dataframe[f'{column}'] / dataframe[f'new_{column}']) - 1 >= 5, 
                                                dataframe[f'new_{column}'], dataframe[f'{column}'])
        
        # Drop the original columns
        dataframe.drop(columns=columns_list, inplace=True)
        
        # Drop the intermediate columns with 'new_' prefix
        dataframe.drop(dataframe.filter(regex='new_').columns, axis=1, inplace=True)
        
        # Rename the corrected columns to their original names
        dataframe.rename(columns=lambda x: reg.sub('_2', '', x), inplace=True)
        
        return dataframe


    def build_historical_data(self, cls_FileHandling, parameters):

        if parameters.execute_filtered:
            crypto_symbols = parameters.filter_symbols
        else:
            # Get a list of cryptocurrency symbols from Kucoin
            kucoin_symbols = self.get_kucoin_symbols()
            # Remove the last character from each symbol in the list
            # USDT to USD - name convention on yfinance
            crypto_symbols = [symbol[:-1] for symbol in kucoin_symbols]

        # Define start date for historical data retrieval
        start_date = parameters.start_date_ingestion

        # Fetch historical cryptocurrency data
        crypto_dataframe = self.fetch_crypto_data(crypto_symbols, start_date)
        crypto_dataframe_treated = self.correcting_numbers_discrepancy(crypto_dataframe)

        if crypto_dataframe_treated is not None:

            # Specify the folder and file name for saving the Parquet file
            output_path = Path(parameters.files_folder, parameters.file_ingestion)

            # Create the output folder if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Removing the data from Index
            crypto_dataframe_treated = crypto_dataframe_treated.reset_index()

            # Save the DataFrame as a Parquet file
            cls_FileHandling.save_parquet_file(crypto_dataframe_treated, output_path)

            print(f"Parquet file saved to {output_path} with {len(crypto_dataframe_treated)} rows")

            cls_FileHandling.wait_for_file(output_path)
            
        else:
            print("No data fetched.")