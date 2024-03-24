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
    # Convert numerical columns to float
    dataframe[['open', 'close', 'high', 'low']] = dataframe[['open', 'close', 'high', 'low']].astype(float)
    
    # Sort DataFrame by 'symbol' and 'date', convert date column to datetime
    dataframe = dataframe.sort_values(by=['symbol', 'date']).reset_index(drop=True)
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Calculate and add Exponential Moving Averages (EMAs)
    for window in [12, 26, 55]:
        dataframe[f'ema_{window}'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.ema(x, window))

    # Calculate and add the minimum and maximum values for a 50-day rolling window
    dataframe['min_50'] = dataframe.groupby('symbol')['close'].transform(lambda x: x.shift(1).rolling(window=50).min())
    
    # Calculate the percentage of risk based on the difference between the close price and the 50-day minimum, relative to the close price, rounded to two decimal places
    dataframe['percent_risk'] = round(((dataframe['close'] - dataframe['min_50']) / dataframe['close']) * 100, 2)

    # Calculate Average Directional Index (ADX)
    dataframe[['vl_adx', 'vl_dmp', 'vl_dmn']] = dataframe.groupby('symbol').apply(lambda x: ta.adx(x['high'], x['low'], x['close'], length=14)).reset_index(drop=True)
    dataframe['nm_adx_trend'] = dataframe['vl_adx'].transform(classify_adx_value)

    # Calculate Relative Strength Index (RSI)
    dataframe['rsi'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.rsi(x))

    # Calculate Ichimoku Cloud indicators
    dataframe[['vl_leading_span_a', 'vl_leading_span_b', 'vl_conversion_line', 'vl_base_line', 'vl_lagging_span']] = dataframe.groupby('symbol').apply(lambda x: ta.ichimoku(x['high'], x['low'], x['close'])[0]).reset_index(drop=True)
    dataframe['vl_price_over_conv_line'] = dataframe['close'] - dataframe['vl_conversion_line']
    dataframe['qt_days_ichimoku_positive'] = count_positive_reset(dataframe['vl_price_over_conv_line'])

    # Calculate the MACD and Signal Line
    dataframe['vl_macd'] = dataframe['ema_12'] - dataframe['ema_26']
    dataframe['vl_macd_signal'] = dataframe.groupby('symbol')['vl_macd'].transform(lambda x: x.ewm(span=9).mean())
    dataframe['vl_macd_delta'] = dataframe['vl_macd'] - dataframe['vl_macd_signal']
    dataframe['qt_days_macd_delta_positive'] = count_positive_reset(dataframe['vl_macd_delta'])

    # Financial Result
    for window in [7, 14]:
        dataframe[f'percent_loss_profit_{window}_days'] = dataframe.groupby('symbol')['close'].transform(lambda x: round((x / x.shift(window) - 1) * 100, 2))

    return dataframe

def filter_indicators_today(dataframe, vl_adx_min=25, vl_macd_delta_min = 0.01, date='2024-02-14'):
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

        # MACD
        (dataframe['qt_days_macd_delta_positive'] >= vl_macd_delta_min) &

        # Price above EMAs
        (dataframe['close'] > dataframe['ema_12']) &
        (dataframe['ema_12'] > dataframe['ema_26']) &
        (dataframe['ema_26'] > dataframe['ema_55'])

    ]

    # Drop unnecessary columns
    columns_to_drop = [
        'open',
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
        'ema_12',
        'ema_26',
        'ema_55',
        'vl_adx',
        'vl_macd',
        'vl_macd_signal',
        'vl_macd_delta'
    ]

    df_indicators = df_indicators.drop(columns=columns_to_drop)
    
    df_indicators.set_index('date', inplace=True)
    return df_indicators

