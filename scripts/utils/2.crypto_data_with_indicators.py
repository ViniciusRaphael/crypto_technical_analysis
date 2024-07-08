import duckdb
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
pd.set_option("display.max_columns", None) 
pd.set_option("display.max_rows", None) 

from indicators_util import add_indicators

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

def save_dataframe_to_parquet(dataframe, file_path):
    """
    Save a Pandas DataFrame as a Parquet file.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame to be saved as a Parquet file.
    - file_path (str): Path where the Parquet file will be saved.
    """
    table = pa.Table.from_pandas(dataframe)
    pq.write_table(table=table, where=file_path, compression='snappy')
    
def main():
    # Specify the input Parquet file path
    input_folder = 'files'
    input_file = 'crypto_historical_data.parquet'
    input_path = Path(input_folder) / input_file

    # Read Parquet file into a Pandas DataFrame
    df = read_parquet_to_dataframe(str(input_path))

    # Add indicators to the DataFrame
    indicators_dataframe = add_indicators(df)

    if indicators_dataframe is not None:
        # Specify the output Parquet file path
        output_folder = 'files'
        output_file = 'crypto_data_with_indicators.parquet'
        output_path = Path(output_folder) / output_file

        # Save the DataFrame with indicators as a Parquet file
        display(indicators_dataframe)
        save_dataframe_to_parquet(indicators_dataframe, output_path)        
        
        print(f"Parquet file with indicators saved to {output_path}")
    else:
        print("No data available.")

if __name__ == "__main__":
    main()
