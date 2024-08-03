from pathlib import Path
from f_models import FileHandling

execute_data_ingestion = True
execute_data_indicators = True
execute_data_prep_models = True

execute_train_models = True
execute_backtest = True
execute_daily_outcome = True

execute_filtered = True


cls_FileHandling = FileHandling()


# Files folder
files_folder = 'files'
file_ingestion = 'crypto_historical_data.parquet'
file_w_indicators = 'crypto_data_with_indicators.parquet'
file_prep_models = 'crypto_data_prep_models.parquet'

path_prep_models = Path(files_folder) / file_prep_models


version_model = 'v1.6'
start_date_backtest = '2024-06-01'
filter_symbols = ['SOL-USD', 'BTC-USD', 'ETH-USD']  # Adicione os símbolos que deseja filtrar

min_volume_prep_models = 250_000
clean_targets_prep_models = True
start_date_ingestion = '2018-01-01' if execute_train_models else '2023-07-01'


folder_trained_models = 'models/trained/'
path_trained_models = str(Path(folder_trained_models))



folder_models = 'models/'

# input_file_logmodels = f'accuracy/log_models_{version_model}.csv'
input_file_logmodels = f'accuracy/log_models.csv'

log_models_path = Path(folder_models) / input_file_logmodels

log_models = FileHandling().read_file(folder_models, input_file_logmodels)



# directory = f'trained/{version_id}/'
directory_models = f'models/trained/{version_model}/'

backtest_path = f'models/results/compound_historical.csv'

daily_outcome_path = 'models/results/proba_scores'



target_list_bol =   [
    # booleans positive
    'bl_target_10P_7d','bl_target_15P_7d','bl_target_20P_7d','bl_target_25P_7d',
    'bl_target_10P_15d','bl_target_15P_15d','bl_target_20P_15d','bl_target_25P_15d', 
    'bl_target_10P_30d','bl_target_15P_30d','bl_target_20P_30d','bl_target_25P_30d',
    # booleans negative
    'bl_target_10N_7d','bl_target_15N_7d','bl_target_20N_7d','bl_target_25N_7d',
    'bl_target_10N_15d','bl_target_15N_15d','bl_target_20N_15d','bl_target_25N_15d', 
    'bl_target_10N_30d','bl_target_15N_30d','bl_target_20N_30d','bl_target_25N_30d' 
]

target_list_val =   [
    # real percentual
    
    'target_7d','target_15d','target_30d'
]

removing_cols = ['Date', 'Symbol', 'Dividends', 'Stock Splits']

remove_target_list = target_list_bol + target_list_val



