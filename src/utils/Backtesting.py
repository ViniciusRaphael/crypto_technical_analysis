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

        # If no symbol is provided, return the entire dataframe
        if symbol is None:
            table = dataframe
        else: 
            # Filter the dataframe based on the provided symbol
            table = dataframe[dataframe['Symbol'] == symbol]
        return table

##### Currently not used
    def specific_crypto_return(self, dataframe, crypto = 'BTC-USD', tabular = True):

        df = self.query_data(dataframe, crypto)

        df.set_index(['Date'], inplace=True)

        prices = df['Close_x']
        entrada = df['final_buy_signal'] == 1
        saida = df['sell_signal'] == 1
        
        pf = vbt.Portfolio.from_signals(close=prices, entries=entrada , exits=saida)    
        
        if tabular == True: #csv
            total_return = pf.total_return()
        else: #imagens
            total_return = pf.plot().show()


        return total_return

    def all_crypto_return(self, dataset):

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

        signals = [f for f in os.listdir(parameters.path_model_signals) if os.path.isfile(os.path.join(parameters.path_model_signals, f))]

        for signal in signals:

            print(f'Running {signal}')

            signals_model = parameters.cls_FileHandling.read_file(parameters.path_model_signals, signal)

            total_return = self.all_crypto_return(signals_model)

            total_return.to_csv(str(parameters.path_model_backtest) + f'/{signal}', index=True)

            # print(self.specific_crypto_return(signals_model, 'SOL-USD', False))
