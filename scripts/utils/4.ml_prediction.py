import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
import duckdb
from pathlib import Path
import pandas as pd


def query_data(dataframe, symbol=None):
    # dataframe = dataframe
    if symbol == None:
        table = duckdb.sql(f"""
                SELECT *
                FROM '{dataframe}'
                    """)
    else: 
        table = duckdb.sql(f"""
                        SELECT *
                        FROM '{dataframe}'
                        WHERE Symbol = '{symbol}'
                            """)
       
    return table.df()

def specific_crypto_return(input_path, crypto = 'BTC'):
    global df
    symbol = crypto + '-USD'

    df = query_data(input_path, symbol)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date'], inplace=True)

    prices = df['Close_x']
    entrada = df['final_buy_signal'] == 1
    saida = df['sell_signal'] == 1
    
    pf = vbt.Portfolio.from_signals(close=prices, entries=entrada , exits=saida)    
    total_return = pf.total_return()
    
    return print(total_return)

def all_crypto_return(input_path):
    global df
    df = query_data(input_path)
    symbol = df['Symbol'].unique()
    result = []
    for count, crypto in enumerate(symbol, start=1):
        print(f'Processing {crypto} ({count} of {len(symbol)})')
        df1 = df.copy()
        df1['Date'] = pd.to_datetime(df1['Date'])
        df1.set_index(['Date'], inplace=True)
        df1 = df1.loc[df1['Symbol'] == crypto]

        prices = df1['Close_x']
        entrada = df1['final_buy_signal'] == 1
        saida = df1['sell_signal'] == 1
        
        try:
            pf = vbt.Portfolio.from_signals(close=prices, entries=entrada, exits=saida)
            stats = pf.stats()
            total_return = pf.total_return()
            
            result.append({
                'crypto': crypto,
                'total_return': total_return,
                **stats.to_dict()
            })
        except Exception as e:
            print(f"An error occurred for {crypto}: {e}")
            continue
        
    final_result = pd.DataFrame(result)
    final_result.sort_values(by=['total_return'], ascending=False, inplace=True)

    return final_result

def main():
    global total_return
    # Specify the input Parquet file path
    input_folder = 'files'
    input_file = 'crypto_indicators_and_signals.parquet'
    input_path = Path(input_folder) / input_file

    total_return = all_crypto_return(input_path)
    return total_return
    
    # return specific_crypto_return(input_path, 'DREAMS')

if __name__ == '__main__':
    main()