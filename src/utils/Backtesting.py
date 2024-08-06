import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
from pathlib import Path
import pandas as pd
import os

class RealBacktest():
    def __init__(self) -> None:
        pass


    def query_data(self, dataframe, symbol=None):
        # dataframe = pd.read_parquet(dataframe)
        # dataframe = pd.read_csv(dataframe)
        # If no symbol is provided, return the entire dataframe
        if symbol is None:
            table = dataframe
        else: 
            # Filter the dataframe based on the provided symbol
            table = dataframe[dataframe['Symbol'] == symbol]
        return table

    def specific_crypto_return(self, dataframe, crypto = 'BTC-USD'):
        # global df
        # symbol = crypto + '-USD'

        df = self.query_data(dataframe, crypto)
        # print(dataframe)
        # df = dataframe[dataframe['Symbol'] == crypto]
        # df['Date'] = pd.to_datetime(df['Date'])
        df.set_index(['Date'], inplace=True)

        prices = df['Close_x']
        entrada = df['final_buy_signal'] == 1
        saida = df['sell_signal'] == 1
        
        pf = vbt.Portfolio.from_signals(close=prices, entries=entrada , exits=saida)    
        total_return = pf.total_return()
        
        return total_return

    def all_crypto_return(self, dataset):
        # global df
        df = self.query_data(dataset)
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

    def backtest_models(self, parameters):
        # global total_return
        # # Specify the input Parquet file path
        # input_folder = 'files'
        # input_file = 'crypto_indicators_and_signals.parquet'
        # input_path = Path(input_folder) / input_file

        signals = [f for f in os.listdir(parameters.path_model_signals) if os.path.isfile(os.path.join(parameters.path_model_signals, f))]

        for signal in signals:

            signals_model = parameters.cls_FileHandling.read_file(parameters.path_model_signals, signal)

            total_return = self.all_crypto_return(signals_model)
            # return total_return

            total_return.to_csv(str(parameters.path_model_backtest) + f'/{signal}', index=True)

            print(self.specific_crypto_return(signals_model, 'SOL-USD'))
            # return self.specific_crypto_return(signals, 'SOL-USD')