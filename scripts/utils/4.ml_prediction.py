import matplotlib.pyplot as plt
import pandas as pd

import vectorbt as vbt

import duckdb
from pathlib import Path
import pandas as pd


def query_data(dataframe, symbol):
    # dataframe = dataframe
    result = duckdb.sql(f"""
                    SELECT *
                    FROM '{dataframe}'
                    WHERE Symbol = '{symbol}'
                        """)
    
    return result.df()

# Specify the input Parquet file path
input_folder = 'files'
input_file = 'crypto_data_with_indicators.parquet'
input_path = Path(input_folder) / input_file

crypto = 'ONT'
symbol = crypto + '-USD'

crypto_data = query_data(input_path, symbol)
crypto_data['Date'] = pd.to_datetime(crypto_data['Date'])
crypto_data.set_index(['Date'], inplace=True)

prices = crypto_data['Close']

entrada =(crypto_data['vl_macd'] > crypto_data['vl_macd_signal']) &\
            (crypto_data['ema_12_above_ema_26'] == True) &\
            (crypto_data['ema_12_above_ema_50'] == True) &\
            (crypto_data['ema_12_above_ema_100'] == True) &\
            (crypto_data['ema_12_above_ema_200'] == True)


saida = (crypto_data['vl_macd'] < crypto_data['vl_macd_signal'])
# (crypto_data['percent_loss_profit_14_days'] < - 0.10)        

            

pf = vbt.Portfolio.from_signals(close=prices, entries=entrada , exits=saida)
profit = pf.total_profit()
# pf.plot_positions().show()
pf.plot().show()


# Symbols = df['Symbol'].unique()
# result = []    
# for count, crypto in enumerate(Symbols, start=1):
#     print(f'Processing {crypto} ({count} of {len(Symbols)})')

#     df1 = df.copy()
#     df1['Date'] = pd.to_datetime(df1['Date'])
#     df1.set_index(['Date'], inplace=True)
#     df1 = df1.loc[(df1['Symbol'] == crypto)]

#     prices = df1['close']

#     entrada = (df1['vl_macd'] > df1['vl_macd_signal']) &\
#                 (df1['vl_macd'] < 0 ) &\
#                 ( df1['qt_days_tendency_positive'] > 0)


#     saida = (df1['vl_macd'] < df1['vl_macd_signal']) &\
#             (df1['percent_loss_profit_14_days'] <= -0.20)
#     try:
#         pf = vbt.Portfolio.from_signals(close=prices, entries=entrada , exits=saida)
#         # profit = pf.total_profit()
#         profit = pf.total_return()

#         result.append({'crypto': crypto,
#                     'result': profit})
#     except:
#         pass
    

# result