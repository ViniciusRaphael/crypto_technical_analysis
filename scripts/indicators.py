# Python standard libraries
from datetime import datetime, timedelta
import time
from configparser import ConfigParser

# Third-party libraries
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sqlalchemy import text

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


def count_positive_reset(df_column):
    """
    Counts consecutive positive values in a DataFrame column and resets count on encountering negative values.

    Args:
        df_column (pandas.Series): The DataFrame column to be processed.

    Returns:
        list: A list containing counts of consecutive positive values.
    """    
    count = 0
    counts = []

    for value in df_column:
        if value > 0:
            count += 1
        else:
            count = 0  # Reset count if a negative value is encountered
        counts.append(count)

    return counts


def add_indicators(dataframe):
    """
    Add technical indicators to the DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing financial data.

    Returns:
    - pd.DataFrame: Updated DataFrame with added technical indicators.
    """    
    # Sort DataFrame by 'symbol' and 'date', convert date column to datetime
    dataframe[['open', 'close', 'high', 'low']] = dataframe[['open', 'close', 'high', 'low']].astype(float)
    # dataframe = dataframe.drop(columns = ['count_duplicate'])
    dataframe = dataframe.sort_values(by=['symbol', 'date']).reset_index(drop=True)
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Calculate and add SMAs
    # for window in [20, 50, 100, 200]:
    #     dataframe[f'ma_{window}'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, window))

    # Calculate and add the minimum and maximum values for a 50-day rolling window
    dataframe['min_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: x.shift(1).rolling(window=50).min())
    dataframe['max_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: x.shift(1).rolling(window=50).max())

    # Calculate ADX
    dataframe[['vl_adx', 'vl_dmp', 'vl_dmn']] = dataframe.groupby('symbol').apply(lambda x: ta.adx(x['high'], x['low'], x['close'], length=14)).reset_index(drop=True)
    dataframe['nm_adx_trend'] = dataframe['vl_adx'].transform(classify_adx_value)

    # Calculate RSI
    dataframe['rsi'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.rsi(x))

    # Calculate Ichimoku Cloud indicators
    dataframe[['vl_leading_span_a', 'vl_leading_span_b', 'vl_conversion_line', 'vl_base_line', 'vl_lagging_span']] = dataframe.groupby('symbol').apply(lambda x: ta.ichimoku(x['high'], x['low'], x['close'])[0]).reset_index(drop=True)
    dataframe['vl_price_over_conv_line'] = dataframe['close'] - dataframe['vl_conversion_line']
    dataframe['qt_days_ichimoku_positive'] = count_positive_reset(dataframe['vl_price_over_conv_line'])
 
    # Calculate MACD
    dataframe[['vl_macd', 'vl_macd_hist', 'vl_macd_signal']]  = dataframe.groupby('symbol')['close'].apply(lambda x: ta.macd(x)).reset_index(drop=True)
    dataframe['vl_macd_delta'] = dataframe['vl_macd'] - dataframe['vl_macd_signal']
    dataframe['qt_days_macd_delta_positive'] = count_positive_reset(dataframe['vl_macd_delta'])


    return dataframe


def filter_indicators_today(dataframe, vl_adx_min = 25, date = '2024-02-14' ,vl_macd_hist_min = 0, vl_macd_delta_min = 0.01, qt_days_supertrend_positive = 1):
    """
    Filters the concatenated DataFrame to select specific indicators for the current date.

    Args:
        dataframe (pandas.DataFrame): Concatenated DataFrame containing processed data.

    Returns:
        pandas.DataFrame: Filtered DataFrame based on specific indicators for the current date.
    """

    df_indicators = dataframe.loc[
        (dataframe['vl_adx'] >= vl_adx_min) &
        (dataframe['date'] == str(date)) &

        # Ichimoku with price above conversion line and base line
        (dataframe['close'] > dataframe['vl_conversion_line']) &
        (dataframe['close'] > dataframe['vl_base_line']) &

        # # vl_macd histogram greater than 0 and signal greater than vl_macd
        (dataframe['vl_macd_hist'] >= vl_macd_hist_min) &
        (dataframe['vl_macd_delta'] >= vl_macd_delta_min) 
        # (dataframe['qt_days_supertrend_positive'] >= qt_days_supertrend_positive)
    ]

        # Drop unnecessary columns
    df_indicators = df_indicators.drop(
        columns = [
            'open',
            'close',
            'high',
            'low',
            'volume',
            'dividends',
            'vl_dmp', 
            'vl_dmn', 
            'vl_leading_span_a', 
            'vl_leading_span_b', 
            'vl_lagging_span',
            'vl_conversion_line', 
            'vl_base_line',
            'vl_price_over_conv_line',
            'vl_macd', 
            'vl_macd_hist', 
            'vl_macd_signal'
            ]
        )

    df_indicators.set_index('date', inplace=True)
    return df_indicators


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

    yesterday_date = datetime.today().date() - timedelta(days=2)
    result = filter_indicators_today(df1, date = yesterday_date)

    