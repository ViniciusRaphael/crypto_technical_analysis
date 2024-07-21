import pandas as pd
import numpy as np
from IPython.display import display # pip install ipython


input_path = r'D:\Github\Forked\crypto_technical_analysis\models\results\compound_historical.csv'

dados_proba = pd.read_csv(input_path)

input_path = r'D:\Github\Forked\crypto_technical_analysis\files\crypto_data_prep_models.parquet'
dados_prep = pd.read_parquet(input_path)

# pd.set_option("display.max_columns", None)

dados_filter = dados_prep[['Symbol', 'Date', 'Volume', 'Close', 'target_10_7d', 'target_10_15d', 'target_10_30d']]
dados_score = dados_proba[['Symbol', 'Date', 'lr_pb_10_7d', 'lr_pb_15_7d', 'lr_pb_20_7d', 'lr_pb_25_7d', 'lr_pb_10_15d', 'lr_pb_15_15d', 'lr_pb_20_15d', 'lr_pb_25_15d', 'lr_pb_10_30d', 'lr_pb_15_30d', 'lr_pb_20_30d', 'lr_pb_25_30d', 'score']]



df_merged = pd.merge(dados_filter, dados_score, on=['Date', 'Symbol'], how='inner')
# Substituindo valores infinitos por NaN
df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
# Removendo linhas com valores NaN
df_merged.dropna(inplace=True)

## Entradas

df_merged_score_clean_cumulative = df_merged[(df_merged['lr_pb_10_15d'] >= 0.7) & (df_merged['Symbol'] == 'NEAR-USD')  & (df_merged['Date'] >= '2024-01-01')]

df_merged_score_clean_cumulative = df_merged_score_clean_cumulative.sort_values(by='Date')


df_merged_score_clean_cumulative['Cumulative_Return_7d'] = (1 + ( df_merged_score_clean_cumulative['target_10_7d'])).cumprod() - 1
df_merged_score_clean_cumulative['Cumulative_Return_15d'] = (1 + ( df_merged_score_clean_cumulative['target_10_15d'])).cumprod() - 1
df_merged_score_clean_cumulative['Cumulative_Return_30d'] = (1 + ( df_merged_score_clean_cumulative['target_10_30d'])).cumprod() - 1


display(df_merged_score_clean_cumulative)#.sort_values(by='Date')
