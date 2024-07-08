# Third-party libraries
import pandas as pd
import pandas_ta as ta



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
    # Sort DataFrame by 'Symbol' and 'Date', convert Date column to Datetime
    dataframe = dataframe.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)

    windows = [12, 26, 50, 100, 200]

    # Calculate and add Exponential Moving Averages (EMAs)
    for window in windows:
        dataframe[f'ema_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.ema(x, window))

    for window in windows[1:]:
        dataframe[f'ema_12_above_ema_{window}'] = dataframe.apply(lambda row: row['ema_12'] > row[f'ema_{window}'] , axis=1)

    # Financial Result
    targets = [10, 15, 20, 25]
    timeframes = [7, 15, 30]

    for target in targets:
        for timeframe in timeframes:
            dataframe[f'target_{target}_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: (1 - (x / x.shift(timeframe))))
            dataframe[f'bl_target_{target}_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: (1 - (x / x.shift(timeframe))) >= target / 100).astype(int)

    # Calculate Average Directional Index (ADX)
    dataframe[['vl_adx', 'vl_dmp', 'vl_dmn']] = dataframe.groupby('Symbol').apply(lambda x: ta.adx(x['High'], x['Low'], x['Close'], length=14)).reset_index(drop=True)
    dataframe['nm_adx_trend'] = dataframe['vl_adx'].transform(classify_adx_value)

    # Calculate Relative Strength Index (RSI)
    dataframe['rsi'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.rsi(x))

    # Calculate the MACD and Signal Line
    dataframe['vl_macd'] = dataframe['ema_12'] - dataframe['ema_26']
    dataframe['vl_macd_signal'] = dataframe.groupby('Symbol')['vl_macd'].transform(lambda x: x.ewm(span=9).mean())
    dataframe['vl_macd_delta'] = dataframe['vl_macd'] - dataframe['vl_macd_signal']
    dataframe['qt_days_macd_delta_positive'] = count_positive_reset(dataframe['vl_macd_delta'])

    return dataframe
