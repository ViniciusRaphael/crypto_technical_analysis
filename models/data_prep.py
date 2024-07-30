import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# input_path = r'/Users/vinicius.sousa/vini_github/crypto_technical_analysis/crypto_data_with_indicators.parquet'

# input_folder = '../scripts/utils/files/'
input_folder = 'files/'

input_file = 'crypto_data_with_indicators.parquet'
input_path = Path(input_folder) / input_file

dados = pd.read_parquet(input_path)

dados['Date'] = pd.to_datetime(dados['Date'])
dados['Date'] = dados['Date'].dt.strftime('%Y-%m-%d')

max_date = str(dados['Date'].max())

active = (dados[dados['Date'] == max_date])
active = active[['Symbol']]

# Filter to clean data
dados_prep = pd.merge(active, dados[(dados['Close'] != 0) & (dados['Volume'] > 250_000) & (dados['target_7d'] < 3) & (dados['target_7d'] > - 0.9) & (dados['target_15d'] < 3) & (dados['target_15d'] > - 0.9) & (dados['target_30d'] < 3) & (dados['target_30d'] > - 0.9)], on=['Symbol'], how='inner')

def save_dataframe_to_parquet(dataframe, file_path):
    """
    Save a Pandas DataFrame as a Parquet file.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame to be saved as a Parquet file.
    - file_path (str): Path where the Parquet file will be saved.
    """
    table = pa.Table.from_pandas(dataframe)
    pq.write_table(table=table, where=file_path, compression='snappy')


# output_folder = '../scripts/utils/files/'
output_folder = 'files/'
output_file = 'crypto_data_prep_models.parquet'
output_path = Path(output_folder) / output_file

# Save the DataFrame with indicators as a Parquet file
# print(indicators_dataframe)
save_dataframe_to_parquet(dados_prep, output_path)        

print(f"Parquet file with indicators prep models saved to {output_path}")