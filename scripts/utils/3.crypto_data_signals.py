import pandas as pd
import duckdb
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

def read_parquet_to_dataframe(parquet_path):
    """
    Read data from a Parquet file into a Pandas DataFrame.

    Parameters:
    - parquet_path (str): Path to the Parquet file.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the Parquet file.
    """
    table = duckdb.read_parquet(parquet_path)
    return table.df()

def crypto_indicators():
    input_folder = 'files'
    input_file = 'crypto_data_with_indicators.parquet'
    input_path = Path(input_folder) / input_file
    crypto_data_w_indicators = read_parquet_to_dataframe(str(input_path))
    return crypto_data_w_indicators

def crypto_signals():
    input_folder = '../../models/'
    input_file = 'results/compound_historical.csv'
    input_path = Path(input_folder) / input_file
    crypto_signals = pd.read_csv(input_path)
    return crypto_signals

def save_dataframe_to_parquet(dataframe, file_path):
    """
    Save a Pandas DataFrame as a Parquet file.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame to be saved as a Parquet file.
    - file_path (str): Path where the Parquet file will be saved.
    """
    table = pa.Table.from_pandas(dataframe)
    pq.write_table(table=table, where=file_path, compression='snappy')

def table_query(crypto_indicators, crypto_signals, model_percentage_cut=0.7):
    crypto_indicators['Date'] = pd.to_datetime(crypto_indicators['Date']).dt.date
    crypto_signals['Date'] = pd.to_datetime(crypto_signals['Date']).dt.date

    # Step 1: Join the tables
    joined_tables = pd.merge(crypto_indicators, crypto_signals, how='left', on=['Symbol', 'Date'])

    # Step 2: Handle missing values
    joined_tables['lr_pb_25_30d'] = joined_tables['lr_pb_25_30d'].fillna(0)

    # Step 3: Calculate initial buy signal
    joined_tables['all_buy_signal'] = np.where(joined_tables['lr_pb_25_30d'] > model_percentage_cut, 1, 0)

    # Step 4: Apply the rule that no two buy signals are within 25 days of each other
    joined_tables['final_buy_signal'] = 0
    last_buy_date = {}
    
    for idx, row in joined_tables.iterrows():
        symbol = row['Symbol']
        if row['all_buy_signal'] == 1:
            if symbol in last_buy_date:
                if (row['Date'] - last_buy_date[symbol]).days > 30:
                    joined_tables.at[idx, 'final_buy_signal'] = 1
                    last_buy_date[symbol] = row['Date']
            else:
                joined_tables.at[idx, 'final_buy_signal'] = 1
                last_buy_date[symbol] = row['Date']

    # Step 5: Create sell signal 25 days after final_buy_signal
    joined_tables['sell_signal'] = 0
    buy_dates = joined_tables[joined_tables['final_buy_signal'] == 1][['Symbol', 'Date']]
    
    for _, buy_row in buy_dates.iterrows():
        sell_date = buy_row['Date'] + pd.Timedelta(days=30)
        sell_idx = joined_tables[(joined_tables['Symbol'] == buy_row['Symbol']) & (joined_tables['Date'] == sell_date)].index
        if not sell_idx.empty:
            joined_tables.at[sell_idx[0], 'sell_signal'] = 1

    return joined_tables


def main():
    global crypto_indicators, crypto_signals, crypto_indicators_and_signals
    crypto_indicators = crypto_indicators()
    crypto_signals = crypto_signals()

    indicators_and_signals_join = table_query(crypto_indicators, crypto_signals, 0.6)

    crypto_indicators_and_signals = indicators_and_signals_join
    output_folder = 'files'
    output_file = 'crypto_indicators_and_signals.parquet'
    output_path = Path(output_folder) / output_file
    
    save_dataframe_to_parquet(crypto_indicators_and_signals, output_path)
    return crypto_indicators_and_signals
if __name__ == "__main__":
    main()
