import pandas as pd

dados = pd.read_parquet('data/crypto_data_with_indicators.parquet')

print(dados.columns)