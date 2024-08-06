import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from datetime import datetime

class Backtesting():
    def __init__(self) -> None:
        pass
        
        
    def table_query(self, parameters, select_model, model_percentage_cut, target_timeframe):
        # crypto_indicators['Date'] = pd.to_datetime(crypto_indicators['Date']).dt.date
        # crypto_signals['Date'] = pd.to_datetime(crypto_signals['Date']).dt.date

        crypto_indicators = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_w_indicators)

        # crypto_signals = parameters.cls_FileHandling.read_file(parameters.file_backtest, '')
        crypto_signals = pd.read_csv(parameters.file_backtest)


        # Step 1: Join the tables
        joined_tables = pd.merge(crypto_indicators, crypto_signals, how='left', on=['Symbol', 'Date'])

        # Step 2: Handle missing values
        joined_tables[select_model] = joined_tables[select_model].fillna(0)

        # Step 3: Calculate initial buy signal
        joined_tables['all_buy_signal'] = np.where(joined_tables[select_model] > model_percentage_cut, 1, 0)

        # Step 4: Apply the rule that no two buy signals are within 25 days of each other
        joined_tables['final_buy_signal'] = 0
        last_buy_date = {}

        formato_data = '%Y-%m-%d' 
        # joined_tables = joined_tables.dropna(axis=1, how='all')

        for idx, row in joined_tables.iterrows():
            symbol = row['Symbol']
            if row['all_buy_signal'] == 1:
                if symbol in last_buy_date:
                  
                    # print(pd.to_datetime(row['Date']) - pd.to_datetime(last_buy_date[symbol]))
                    
                    if (datetime.strptime(row['Date'], formato_data) - datetime.strptime(last_buy_date[symbol], formato_data)).days > int(target_timeframe):

                        joined_tables.at[idx, 'final_buy_signal'] = 1
                        last_buy_date[symbol] = row['Date']
                else:
                    joined_tables.at[idx, 'final_buy_signal'] = 1
                    last_buy_date[symbol] = row['Date']


        # Step 5: Create sell signal 25 days after final_buy_signal
        joined_tables['sell_signal'] = 0
        buy_dates = joined_tables[joined_tables['final_buy_signal'] == 1][['Symbol', 'Date']]
        pd.to_datetime(row['Date'])
        for _, buy_row in buy_dates.iterrows():
            sell_date =  pd.to_datetime(buy_row['Date']) + pd.Timedelta(days=int(target_timeframe))
            sell_idx = joined_tables[(joined_tables['Symbol'] == buy_row['Symbol']) & (joined_tables['Date'] == sell_date)].index
            if not sell_idx.empty:
                joined_tables.at[sell_idx[0], 'sell_signal'] = 1

        return joined_tables


    def build_signal_model(self, parameters, select_model, model_percentage_cut=0.7, target_timeframe=30):

        crypto_indicators_and_signals = self.table_query(parameters, select_model, model_percentage_cut, target_timeframe)

        crypto_indicators_and_signals.to_csv(str(parameters.path_model_signals) + f'/{parameters.version_model}_{select_model}_{model_percentage_cut}', index=True)


        # parameters.cls_FileHandling.save_parquet_file(crypto_indicators_and_signals, str(Path(parameters.path_model_signals)) + f'_{select_model}_{model_percentage_cut}_.csv')

        return crypto_indicators_and_signals
    
    
    def build_signals(self, parameters):

        log_models = parameters.cls_Predict.choose_best_models(parameters)

        name_models = list(set(log_models['name_model']))

        print(name_models)

        for id_model in name_models:
            target_time_frame_select = id_model.split('_')[-1][:-1]

            print(f'Build signals for {id_model} at timeframe {target_time_frame_select}')

            self.build_signal_model(parameters, id_model, parameters.min_threshold_signals, target_time_frame_select)

