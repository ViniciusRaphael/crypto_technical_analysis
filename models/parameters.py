from pathlib import Path
import pandas as pd
from f_models import DataPrep, FileHandling
import os

# input_folder = '../scripts/utils/files/'
input_folder = 'files'
input_file = 'crypto_data_with_indicators.parquet'
file_w_indicators = 'crypto_data_with_indicators.parquet'
file_ingestion = 'crypto_historical_data.parquet'


input_path = Path(input_folder) / input_file
# dados0 = pd.read_parquet(input_path)
dados0 = FileHandling().read_file(input_folder, input_file)

# if os.path.exists(input_path) and os.stat(input_path).st_size > 0:
#     dados0 = pd.read_csv(input_path)

# Suponha que 'dados_prep_models' seja o seu DataFrame


execute_data_ingestion = True
execute_data_indicators = True
execute_data_prep_models = True

execute_train_models = True
execute_backtest = False
execute_daily_outcome = True

execute_filtered = True


version_model = 'v1.5'
start_date_backtest = '2024-06-01'
# start_date_ingestion = '2018-01-01'

start_date_ingestion = '2018-01-01' if execute_train_models else '2023-07-01'



filter_symbols = ['SOL-USD', 'BTC-USD', 'ETH-USD']  # Adicione os símbolos que deseja filtrar

# dados_indicators_filtered =  DataPrep().clean_date(dados0[dados0['Symbol'].isin(filter_symbols)])
# dados_indicators_all =  DataPrep().clean_date(dados0)
# dados_indicators = dados_indicators_filtered if execute_filtered else dados_indicators_all
dados_indicators = dados0




# input_folder = '../scripts/utils/files/'
save_in_folder = 'models/trained/'
trained_models_path = str(Path(save_in_folder))


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

## vai gerar a pasta em models/trained/versao se a versa já estiver criada, sobrescreve, se não, cria a pasta


# input_folder = '../scripts/utils/files/'
# input_folder = 'files/'

input_file_prep_models = 'crypto_data_prep_models.parquet'
input_path_prep_models = Path(input_folder) / input_file_prep_models
# dados_prep_models0 = pd.read_parquet(input_path_prep_models)

dados_prep_models0 = FileHandling().read_file(input_folder, input_file_prep_models)



# dados_prep_models_filtered = dados_prep_models0[dados_prep_models0['Symbol'].isin(filter_symbols)]
# dados_prep_models_all = dados_prep_models0
# dados_prep_models = dados_prep_models_filtered if execute_filtered else dados_prep_models_all
dados_prep_models = dados_prep_models0




min_volume_prep_models = 250_000
clean_targets_prep_models = True

# input_folder = '../models/'
input_folder_models = 'models/'

# input_file_logmodels = f'accuracy/log_models_{version_model}.csv'
input_file_logmodels = f'accuracy/log_models.csv'

log_models_path = Path(input_folder_models) / input_file_logmodels

# log_models = pd.read_csv(log_models_path)

log_models = FileHandling().read_file(input_folder_models, input_file_logmodels)

# directory = f'trained/{version_id}/'
directory_models = f'models/trained/{version_model}/'


# if os.path.exists(log_models_path) and os.stat(log_models_path).st_size > 0:
#     log_models = pd.read_csv(log_models_path)
# else:
#     log_models = pd.DataFrame(columns=['name_file', 'name_model', 'target', 'version', 'date_add', 'true_negative', 'false_positive', 'false_negative', 'true_positive', 'accuracy', 'precision', 'recall', 'auc_roc', 'f1_score'])


# log_models = pd.read_csv(log_models_path)

backtest_path = f'models/results/compound_historical.csv'

daily_outcome_path = 'models/results/proba_scores'