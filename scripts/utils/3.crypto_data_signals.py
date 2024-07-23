import pandas as pd
import duckdb
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

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

def table_query(model_percentage_cut=0.7):
    table = duckdb.sql(
        f"""
        WITH join_tables AS (
            SELECT 
                ci.*,
                COALESCE(lr_pb_25_30d, 0) AS lr_pb_25_30d
            FROM crypto_indicators AS ci
            LEFT JOIN crypto_signals AS cs
            ON ci.Symbol = cs.Symbol
            AND CAST(ci.Date AS DATE) = CAST(cs.Date AS DATE)
            ),
        
        buy_signal_logic AS (
            SELECT 
                *,   
                IF(lr_pb_25_30d > {model_percentage_cut},
                    1, 
                    0
                ) AS all_buy_signal
            FROM join_tables
        ),

        remove_sequential_buy_signals AS (
            SELECT * EXCLUDE(all_buy_signal),
            IF(
                all_buy_signal = 1
                AND SUM(all_buy_signal) OVER (PARTITION BY Symbol ORDER BY Date ASC ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) = 0,
                1,
                0
            ) AS buy_signal
            FROM buy_signal_logic
        )

        SELECT 
            *,
            COALESCE(
                LAG(buy_signal, 25) OVER (PARTITION BY Symbol ORDER BY Date ASC),
                0
            ) AS sell_signal
        FROM remove_sequential_buy_signals         
        """
    )

    return table

def main():
    global crypto_indicators, crypto_signals, crypto_indicators_and_signals
    crypto_indicators = crypto_indicators()
    crypto_signals = crypto_signals()

    indicators_and_signals_join = table_query(0.6)

    crypto_indicators_and_signals = indicators_and_signals_join.df()
    output_folder = 'files'
    output_file = 'crypto_indicators_and_signals.parquet'
    output_path = Path(output_folder) / output_file
    
    save_dataframe_to_parquet(crypto_indicators_and_signals, output_path)
    return crypto_indicators_and_signals
if __name__ == "__main__":
    main()
