# import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
import pandas as pd
import os
import numpy as np
import re


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


        root_path_version = str(parameters.path_model_backtest) + '//' + str(parameters.version_model)
        if not os.path.exists(root_path_version):
            # Cria a pasta
            os.makedirs(root_path_version)

        for signal in signals:

            print(f'Running {signal}')

            signals_model = parameters.cls_FileHandling.read_file(parameters.path_model_signals, signal)

            total_return = self.all_crypto_return(signals_model)

            total_return.to_csv(root_path_version + f'/{signal}', index=True, sep=';', decimal=',')

            print(f'Backtest saved in {root_path_version}/{signal}')


    def reached_target(self, dataset, signal_suffix, target_suffix, percent_suffix):
        
        target_col = 'target_' + target_suffix + 'd'

        if signal_suffix == 'P': 
            dataset['reached_target'] = dataset[target_col] >= int(percent_suffix)/100

        elif signal_suffix == 'N':
            dataset['reached_target'] = dataset[target_col] <= - int(percent_suffix)/100
        
        else:
            raise Exception('Invalid signal_suffix [P or N]')

        return dataset
    
    
    def simulate_return_value(self, dataset, target_suffix):
        target_col = 'target_' + target_suffix + 'd'
        entry_value_invest = 100

        dataset['simulate_entry'] = entry_value_invest 
        dataset['simulate_return'] = entry_value_invest * (1 + dataset[target_col]) # Set the return of a given entry value

        return dataset
    

    def reached_target_row(self, row):
        # Extrair o número embutido na variável
        match = re.search(r'_(\d+)([PN])_', row['model'])

        if match:
            valor_embutido = int(match.group(1))  # Mantém o valor original para comparação
            tipo = match.group(2)
            
            # Converte a percentagem para a escala correta
            valor_percentagem = row['simulate_variation']
            
            if tipo == 'P':
                return valor_percentagem >= int(valor_embutido)/100
            elif tipo == 'N':
                return valor_percentagem <= -int(valor_embutido)/100
            
        return None  # Caso não tenha correspondência (não deve acontecer)


    def all_entries_backtest(self, parameters):
        # Read in the necessary files
        predicted_backtest = pd.read_csv(parameters.file_backtest)
        dados_prep_models = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_prep_models)

        # Filter and merge data
        dados_filter = dados_prep_models[['Symbol', 'Date', 'Volume', 'target_7d', 'target_15d', 'target_30d']]
        df_merged = pd.merge(dados_filter, predicted_backtest, on=['Date', 'Symbol'], how='inner')

        # Filter for dates and threshold only once
        df_merged = df_merged[df_merged['Date'] >= parameters.start_date_backtest]
        log_models = parameters.cls_Predict.choose_best_models(parameters)

        models = list(set(log_models['name_model']))

        # Prepare columns for results
        result_columns = ['Symbol', 'model', 'number_correct_entries', 'number_entries', 'percent_correct_entries', 'simulate_entry', 'simulate_return', 'simulate_variation', 'first_entry', 'last_entry']
        backtest_dataset_return = []

        unique_cryptos = df_merged['Symbol'].unique()
        count = 1

        for crypto in unique_cryptos:
            print(f'Processing {crypto} ({count} of {len(unique_cryptos)})')

            # Filter once per crypto
            crypto_df = df_merged[df_merged['Symbol'] == crypto]
            for col in models:
                signal_df = crypto_df[['Symbol', 'Date', 'Volume', 'target_7d', 'target_15d', 'target_30d','Close', col]]

                signal_df = signal_df[crypto_df[col] >= parameters.min_threshold_signals]

                if signal_df.empty:
                    continue

                # Sort values by Date
                signal_df = signal_df.sort_values(by='Date')

                # Verify if the target was reached
                signal_suffix = col.split('_')[-2][-1]
                target_suffix = col.split('_')[-1][:-1]
                percent_suffix = col.split('_')[-2][:-1]

                signal_df = self.reached_target(signal_df, signal_suffix, target_suffix, percent_suffix)


                # Simulate a value return with an arbitrary value
                signal_df = self.simulate_return_value(signal_df, target_suffix)
                cumulative_return = {
                    'Symbol': crypto,
                    'model': col,
                    'number_correct_entries': sum(signal_df['reached_target']),
                    'number_entries': len(signal_df),
                    'percent_correct_entries': sum(signal_df['reached_target'])/len(signal_df), 
                    'simulate_entry': sum(signal_df['simulate_entry']),
                    'simulate_return': sum(signal_df['simulate_return']),
                    'simulate_variation': (sum(signal_df['simulate_return']) - sum(signal_df['simulate_entry'])) / sum(signal_df['simulate_entry']),
                    'first_entry': signal_df['Date'].min(),
                    'last_entry': signal_df['Date'].max()
                }

                backtest_dataset_return.append(cumulative_return)

            count += 1
        
        # Convert list of results to DataFrame and save
        backtest_dataset_return_df = pd.DataFrame(backtest_dataset_return, columns=result_columns)
        backtest_dataset_return_df['reached_target'] = backtest_dataset_return_df.apply(self.reached_target_row, axis=1)        
        
        daily_output_filename = f'{parameters.path_model_backtest}/_simple_backtest_{parameters.version_model}_{parameters.min_threshold_signals}_.csv'

        backtest_dataset_return_df.to_csv(daily_output_filename, index=True, sep=';', decimal=',')
        
        print(f'Daily predict saved in {daily_output_filename}')

        return backtest_dataset_return_df
    

    def simulator(self, parameters):
        # Read in the necessary files
        predicted_backtest = pd.read_csv(parameters.file_backtest)
        dados_prep_models = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_prep_models)

        # Filter and merge data
        dados_filter = dados_prep_models[['Symbol', 'Date', 'Volume', 'target_7d', 'target_15d', 'target_30d']]
        df_merged = pd.merge(dados_filter, predicted_backtest, on=['Date', 'Symbol'], how='inner')

        # Filter for dates and threshold only once
        df_merged = df_merged[df_merged['Date'] >= parameters.start_date_backtest]
        log_models = parameters.cls_Predict.choose_best_models(parameters)

        models = list(set(log_models['name_model']))

        # Prepare columns for results
        result_columns = ['simulation', 'model', 'number_correct_entries', 'number_entries', 'percent_correct_entries', 'simulate_entry', 'simulate_return', 'simulate_variation', 'cryptos_in_simulation']

        simulation_dataset_return_compiled = []

        count = 1

        for col in models:
            
            print(f'Processing simulations for {col} ({count} of {len(models)})')

            signal_df = df_merged[['Symbol', 'Volume', 'target_7d', 'target_15d', 'target_30d','Close', col]]

            signal_df = signal_df[signal_df[col] >= parameters.min_threshold_signals]

            c_simulation = 1
            # simulation_dataset_return = []

            for simulation in range(1, parameters.numbers_of_simulations + 1, 1):
                
                # print(f'Processing simulation {simulation} ({c_simulation} of {parameters.numbers_of_simulations})')

                signal_simulation = signal_df

                signal_simulation['random_number'] = np.random.randint(0, 10_000_000, size=len(signal_simulation))

                signal_simulation = signal_simulation.sort_values(by='random_number').head(parameters.numbers_of_entries_day_simulations)

                if signal_simulation.empty:
                    continue

                # Verify if the target was reached
                signal_suffix = col.split('_')[-2][-1]
                target_suffix = col.split('_')[-1][:-1]
                percent_suffix = col.split('_')[-2][:-1]

                signal_simulation = self.reached_target(signal_simulation, signal_suffix, target_suffix, percent_suffix)

                # Simulate a value return with an arbitrary value
                signal_simulation = self.simulate_return_value(signal_simulation, target_suffix)
                cumulative_return = {
                    'simulation': c_simulation,
                    # 'Date': date_select,
                    'model': col,
                    'number_correct_entries': sum(signal_simulation['reached_target']),
                    'number_entries': len(signal_simulation),
                    'percent_correct_entries': sum(signal_simulation['reached_target'])/len(signal_simulation), 
                    'simulate_entry': sum(signal_simulation['simulate_entry']),
                    'simulate_return': sum(signal_simulation['simulate_return']),
                    'simulate_variation': (sum(signal_simulation['simulate_return']) - sum(signal_simulation['simulate_entry'])) / sum(signal_simulation['simulate_entry']),
                    'cryptos_in_simulation': set(signal_simulation['Symbol'].unique()) if parameters.return_crypto_in_simulations else ''
                }

                simulation_dataset_return_compiled.append(cumulative_return)

                c_simulation += 1
                
            count += 1

        # Convert list of results to DataFrame and save
        simulation_dataset = pd.DataFrame(simulation_dataset_return_compiled, columns=result_columns)
        
        simulation_output_filename = f'{parameters.path_model_simulations}_simulation_{parameters.min_threshold_signals}_{parameters.numbers_of_simulations}_{parameters.numbers_of_entries_day_simulations}.csv'

        simulation_dataset.to_csv(simulation_output_filename, index=False, sep=';', decimal=',')
        
        print(f'Simulations saved in {simulation_output_filename}')

        return simulation_dataset