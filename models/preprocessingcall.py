# import preprocessing
from pathlib import Path
import pandas as pd

input_folder = ''
input_file = 'crypto_data_with_indicators.parquet'
input_path = Path(input_folder) / input_file

dados = pd.read_parquet('crypto_data_with_indicators.parquet')

print(dados)