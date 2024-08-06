from pathlib import Path
# from src.utils.f_models import FileHandling
from src.utils.FileHandle import FileHandling
from src.utils.Transform import DataTransform
from src.utils.Train import Models
from src.utils.PrepModels import DataPrep
from src.utils.Predict import Deploy
from src.utils.Ingestion import DataIngestion
from src.utils.Signals import Backtesting
from src.utils.Features import Features
from src.utils.Backtesting import RealBacktest









# Process selections
execute_data_ingestion = True               # It will play the ingestion pipeline
execute_data_indicators = True              # It will play the indicators pipeline
execute_data_prep_models = True             # It will play the data prep models pipeline (used to train the model)
execute_train_models = False                 # It will play the train models pipeline 
execute_historical_predict = False                     # It will play the backtest pipeline, for futher scenarios validation
execute_daily_predict = True                # It will play the daily outcome pipeline, default is the last recent, but you can set another date in enrichment file
execute_filtered = False                     # It will filter symbols by the filter_symbols
execute_signals = True        
execute_backtest = True             


# Configs scores and model version
score_metric = 'precision'                   # Metric to compose the score. Options: accuracy, precision, recall, auc_roc, f1_score
version_model = 'v1.1'                       # Define the version. If it doesnt exist, it will be created (when trained the model) otherwise, it will used the previously one
num_select_models = 10       # select the max number of models to return (0 for fall)
min_threshold_models = 0  # select the minimum threshold for select the model (considering the score_metric)


# Configs data filters
filter_symbols = ['SOL-USD', 'BTC-USD', 'ETH-USD']  # Filter symbols only when the execute_filtered is True
start_date_backtest = '2024-07-01'                  # Define the start date for backtesting
start_date_ingestion = '2018-01-01' if execute_train_models else '2023-07-01'  # We only need data for the last 200 days for daily_outcome, but we need the historical for training


# Configs training variables
min_volume_prep_models = 250_000    # Define the minimum daily volume that must be considered when training
clean_targets_prep_models = True    # If True, remove outliers when training (beta)
removing_cols_for_train = ['Date', 'Symbol', 'Dividends', 'Stock Splits']      # Removing cols when training and predict (the model that you use my have the same config)



####################################################################
# Auxiliary definitions 

cls_FileHandling = FileHandling()
# cls_File = FileHandling()
cls_Models = Models()
cls_Predict = Deploy()
cls_DataPrep = DataPrep()
cls_Ingestion = DataIngestion()
cls_Transform = DataTransform()
cls_Signals = Backtesting()
cls_Features = Features()
cls_RealBacktest = RealBacktest()



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


path_model_signals = Path(f'output/signals/')
min_threshold_signals = 0.7


path_model_backtest = Path(f'output/backtest/')


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