# Python standard libraries
from configparser import ConfigParser

# Third-party libraries
import pandas as pd
import pandas_ta as ta
from sqlalchemy import text
import plotly.graph_objects as go
import time
import numpy as np

# My own libraries or custom modules
import db_util as du

# Read configuration from 'config.ini' file
config = ConfigParser()
config.read('../config.ini')

def classify_adx_value(value):
    """
    Checks the ADX value against predefined ranges and returns the corresponding trend category.

    Args:
        value (int): The ADX value to be categorized.

    Returns:
        str or None: The category name for the provided ADX value.
    """    
    NM_ADX_GROUP = {
    '0-25': '1. Absent or Weak Trend',
    '25-50': '2. Strong Trend',
    '50-75': '3. Very Strong Trend',
    '75-100': '4. Extremely Strong Trend'
    }

    for key, name in NM_ADX_GROUP.items():
        range_start, range_end = map(int, key.split('-'))
        if range_start <= value <= range_end:
            return name
    return None

def add_indicators(dataframe):
    """
    Add indicators to the DataFrame, including the 21-day Simple Moving Average (SMA) and the maximum of the last 55 days.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing financial data.

    Returns:
    - pd.DataFrame: The updated DataFrame with added indicators.
    """

    # Sort the DataFrame by 'symbol' and 'date' to ensure proper application of operations
    # Convert certain columns to float
    dataframe[['open', 'close', 'high', 'low']] = dataframe[['open', 'close', 'high', 'low']].astype(float)

    # Sort DataFrame by 'symbol' and 'date' columns
    dataframe = dataframe.sort_values(by=['symbol', 'date'])

    # Convert the 'date' column to datetime format
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Calculate and add the x-day Simple Moving Average (SMA) to the DataFrame
    dataframe['ma_20'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 20))
    dataframe['ma_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 50))
    dataframe['ma_100'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 100))
    dataframe['ma_200'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 200))

    # Calculate and add the minimum and maximum values for a 50-day rolling window
    dataframe['min_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: x.shift(1).rolling(window=50).min())
    dataframe['max_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: x.shift(1).rolling(window=50).max())

    # Calculate the ADX (Average Directional Movement Index) for each symbol
    # Reset the index to drop unnecessary grouping index
    dataframe[['vl_adx', 'vl_dmp', 'vl_dmn']] = dataframe.groupby('symbol').apply(
        lambda x: ta.adx(x['high'], x['low'], x['close'], length=14)
    ).reset_index(drop=True)

    # Classify the ADX values into trend categories
    dataframe['nm_adx_trend'] = dataframe['vl_adx'].transform(classify_adx_value)

    # Calculate the Relative Strength Index (RSI) for each symbol
    dataframe['rsi'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.rsi(x))

    # Drop unnecessary columns 'vl_dmp' and 'vl_dmn'
    dataframe.drop(columns=['vl_dmp', 'vl_dmn'])
    return dataframe


# Configuration parameters for the PostgreSQL database
db_connection = {
    'host': config.get('Database', 'host'),
    'port': config.getint('Database', 'port'),
    'database': config.get('Database', 'database'),
    'user': config.get('Database', 'user'),
    'password': config.get('Database', 'password')
}   

# Specify the name of the table in the database
table_name = 'crypto_historical_price'

engine = du.connect_to_database(db_connection)

# Use a context manager to handle the connection and automatically close it when done
with engine.connect() as conn:
    df = du.get_db_data(conn, table_name)

    # Ordenar o DataFrame por 'Nome do Grupo' e 'Data'
    start = time.time()
    df1 = add_indicators(df)
    end = time.time()
    print(f'Code finished in: {end - start} sec')

# result = df1.loc[ 
#     (df1['date'] == '2024-02-12')
#     & (df1['close'] > df1['ma_20'])
#     & (df1['close'] > df1['ma_50'])
#     & (df1['close'] > df1['ma_100'])
#     & (df1['close'] > df1['ma_200'])
#     & (df1['ma_20'] > df1['ma_50'])
#     & (df1['ma_50'] > df1['ma_100'])
#     & (df1['ma_100'] > df1['ma_200'])]
