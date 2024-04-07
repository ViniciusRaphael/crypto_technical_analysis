import duckdb
from pathlib import Path

from indicators_util import add_indicators
import pyarrow as pa
import pyarrow.parquet as pq

output_folder = 'files'
output_file = 'crypto_historical_data.parquet'
output_path = str(Path(output_folder) / output_file)

# duckdb.read_parquet(output_path)

df = duckdb.read_parquet(output_path).df()


indicators_dataframe = add_indicators(df)

def save_parquet_file(dataframe, file_path):
    table = pa.Table.from_pandas(dataframe)
    pq.write_table(table=table, where=file_path, compression='snappy')


if indicators_dataframe is not None:
    # Specify the folder and file name for saving the Parquet file
    output_folder = 'files'
    output_file = 'crypto_data_with_indicators.parquet'
    output_path = Path(output_folder) / output_file

    # Create the output folder if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame as a Parquet file
    save_parquet_file(indicators_dataframe, output_path)

    print(f"Parquet file saved to {output_path}")