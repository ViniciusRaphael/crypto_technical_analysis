import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
from pathlib import Path
import pandas as pd
import os
import numpy as np

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

        prices = df['Close']
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

            prices = df1['Close']
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

    def all_entries_backtest(self, parameters):
    
        predicted_backtest = pd.read_csv(parameters.file_backtest)
    
        dados_prep_models = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_prep_models)


        dados_filter = dados_prep_models[['Symbol', 'Date', 'Volume', 'target_7d', 'target_15d', 'target_30d']]


        df_merged = pd.merge(dados_filter, predicted_backtest, on=['Date', 'Symbol'], how='inner')
        # Substituindo valores infinitos por NaN
        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Removendo linhas com valores NaN
        df_merged.dropna(inplace=True)

        ## Entradas
        backtest_dataset_return = pd.DataFrame()
        count = 1
        for crypto in df_merged['Symbol'].unique():
            print(f'Processing {crypto} ({count} of {len(df_merged['Symbol'].unique())})')

            for col in df_merged.columns:
                if (col[-1] == 'd' and col.split('_')[0] != 'target'): 

                    df_selected_cols = df_merged[['Date', 'Symbol', 'target_7d', 'target_15d', 'target_30d', col]]

                    df_merged_score_clean_cumulative = df_selected_cols[(df_selected_cols[col] >= parameters.min_threshold_signals) & (df_selected_cols['Symbol'] == crypto)  & (df_selected_cols['Date'] >= parameters.start_date_backtest)]

                    df_merged_score_clean_cumulative = df_merged_score_clean_cumulative.sort_values(by='Date')

                    df_merged_score_clean_cumulative['Cumulative_Return_7d'] = (1 + ( df_merged_score_clean_cumulative['target_7d'])).cumprod() - 1
                    df_merged_score_clean_cumulative['Cumulative_Return_15d'] = (1 + ( df_merged_score_clean_cumulative['target_15d'])).cumprod() - 1
                    df_merged_score_clean_cumulative['Cumulative_Return_30d'] = (1 + ( df_merged_score_clean_cumulative['target_30d'])).cumprod() - 1


                    df_merged_score_clean_cumulative.drop(['target_7d', 'target_15d', 'target_30d'], axis=1, inplace=True)

                    cumulative_return = df_merged_score_clean_cumulative.tail(1)#.sort_values(by='Date')

                    cumulative_return['model'] = col
                    cumulative_return['number_entries'] = len(df_merged_score_clean_cumulative)
                    cumulative_return['first_entry'] = df_merged_score_clean_cumulative['Date'].min()
                    cumulative_return['last_entry'] = df_merged_score_clean_cumulative['Date'].max()

                    backtest_dataset_return = pd.concat([backtest_dataset_return, cumulative_return])
                
            count += 1
        
        backtest_dataset_return.to_csv(str(parameters.path_model_backtest) + f'/_simple_backtest_{parameters.min_threshold_signals}_.csv', index=True)

        print(f'File saved in {str(parameters.path_model_backtest)}/_simple_backtest_{parameters.min_threshold_signals}_.csv')

        return backtest_dataset_return