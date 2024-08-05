from pathlib import Path
from src.utils.f_models import FileHandling


# Process selections
execute_data_ingestion = False               # If True, it will play the ingestion pipeline
execute_data_indicators = False              # If True, it will play the indicators pipeline
execute_data_prep_models = False             # If True, it will play the data prep models pipeline (used to train the model)
execute_train_models = False                 # If True, it will play the train models pipeline 
execute_backtest = True                     # If True, it will play the backtest pipeline, for futher scenarios validation
execute_daily_outcome = True                # If True, it will play the daily outcome pipeline, default is the last recent, but you can set another date in enrichment file
execute_filtered = False                     # If True, it will filter symbols by the filter_symbols


# Configs scores and model version
score_metric = 'precision'                   # Metric to compose the score. Options: accuracy, precision, recall, auc_roc, f1_score
version_model = 'v1.0'                       # Define the version. If it doesnt exist, it will be created (when trained the model) otherwise, it will used the previously one
num_select_models = 10       # select the max number of models to return (0 for fall)
min_threshold_models = 0.45  # select the minimum threshold for select the model (considering the score_metric)


# Configs data filters
filter_symbols = ['SOL-USD', 'BTC-USD', 'ETH-USD']  # Filter symbols only when the execute_filtered is True
start_date_backtest = '2024-06-01'                  # Define the start date for backtesting
start_date_ingestion = '2018-01-01' if execute_train_models else '2023-07-01'  # We only need data for the last 200 days for daily_outcome, but we need the historical for training


# Configs training variables
min_volume_prep_models = 250_000    # Define the minimum daily volume that must be considered when training
clean_targets_prep_models = True    # If True, remove outliers when training (beta)
removing_cols_for_train = ['Date', 'Symbol', 'Dividends', 'Stock Splits']      # Removing cols when training and predict (the model that you use my have the same config)



####################################################################
# Auxiliary definitions 

cls_FileHandling = FileHandling()


#  Constants (Recomend: Do not change it. Except if your change will add or remove any of them)
####################################################################

# Path Folders and Files (Do not change)
files_folder = 'data'
file_ingestion = 'crypto_data_historical.parquet'
file_w_indicators = 'crypto_data_with_indicators.parquet'
file_prep_models = 'crypto_data_prep_models.parquet'


file_log_models = Path('output/accuracy') / f'log_model_{version_model}.csv'
path_models = Path(f'output/models/{version_model}')
file_backtest = Path(f'output/predict/compound_backtest.csv')
path_daily_outcome = Path('output/predict/proba_scores')

####################################################################

# Targets parameters (Do not change: Or the model can suffer target leakage)

target_list_bol = [
    # booleans positive
    'bl_target_10P_7d','bl_target_15P_7d','bl_target_20P_7d','bl_target_25P_7d',
    'bl_target_10P_15d','bl_target_15P_15d','bl_target_20P_15d','bl_target_25P_15d', 
    'bl_target_10P_30d','bl_target_15P_30d','bl_target_20P_30d','bl_target_25P_30d',
    # booleans negative
    'bl_target_10N_7d','bl_target_15N_7d','bl_target_20N_7d','bl_target_25N_7d',
    'bl_target_10N_15d','bl_target_15N_15d','bl_target_20N_15d','bl_target_25N_15d', 
    'bl_target_10N_30d','bl_target_15N_30d','bl_target_20N_30d','bl_target_25N_30d' 
    ]

# real target
target_list_val =   ['target_7d','target_15d','target_30d']

# removing targets for train the models
remove_target_list = target_list_bol + target_list_val