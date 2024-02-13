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


def add_indicators(dataframe):
    """
    Add indicators to the DataFrame, including the 21-day Simple Moving Average (SMA) and the maximum of the last 55 days.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing financial data.

    Returns:
    - pd.DataFrame: The updated DataFrame with added indicators.
    """

    # Sorts the DataFrame by 'symbol' and 'date' to ensure proper application of operations
    dataframe = dataframe.sort_values(by=['symbol', 'date'])
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Calculates and adds the x-day Simple Moving Average (SMA) to the DataFrame
    dataframe['ma_20'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 20))
    dataframe['ma_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 50))
    dataframe['ma_100'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 100))
    dataframe['ma_200'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 200))

    dataframe['min_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: x.shift(1).rolling(window=50).min())
    dataframe['max_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: x.shift(1).rolling(window=50).max())

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
