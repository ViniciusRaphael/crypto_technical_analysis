import pandas as pd

# dados = pd.read_parquet('data/crypto_data_with_indicators.parquet')
dados = pd.read_parquet('data/crypto_data_prep_models.parquet')
# dados = pd.read_parquet('data/crypto_data_historical.parquet')

# dados = pd.read_csv('output/predict/compound_backtest.csv')


# 
# print(dados.columns)
# print(dados)
# print(len(dados))
# print(dados['Date'].max())
# print(dados['Date'].min())
# print(dados.dropna())
# print(dados['Symbol'].value_counts(),)

crypto = dados[(dados['Symbol'] == 'PENDLE-USD')]
crypto_date = crypto[crypto['Date'] == '2024-07-10']
print(crypto_date.tail(30))
print(crypto_date)
print(dados)