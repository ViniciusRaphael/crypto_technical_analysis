from pathlib import Path
import pandas as pd

# input_folder = '../scripts/utils/files/'
input_folder = 'files/'
input_file = 'crypto_data_prep_models.parquet'
input_path = Path(input_folder) / input_file
dados0 = pd.read_parquet(input_path)

dados_indicators = dados0[dados0['Symbol'] == 'SOL-USD']

# input_folder = '../scripts/utils/files/'
save_in_folder = 'models/trained/'
root_path = str(Path(save_in_folder))


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


## vai gerar a pasta em models/trained/versao se a versa já estiver criada, sobrescreve, se não, cria a pasta
version_model = 'v1.5'

remove_target_list = target_list_bol + target_list_val


# input_folder = '../scripts/utils/files/'
# input_folder = 'files/'

input_file_prep_models = 'crypto_data_prep_models.parquet'
input_path_prep_models = Path(input_folder) / input_file_prep_models
dados_prep_models = pd.read_parquet(input_path_prep_models)

dados_prep_models = dados_prep_models[dados_prep_models['Symbol'] == 'SOL-USD']



# input_folder = '../models/'
input_folder_models = 'models/'

input_file_logmodels = 'accuracy/log_models.csv'
log_models_path = Path(input_folder_models) / input_file_logmodels



# directory = f'trained/{version_id}/'
directory_models = f'models/trained/{version_model}/'


log_models = pd.read_csv(log_models_path)

backtest_path = f'models/results/compound_historical.csv'

daily_outcome_path = 'models/results/proba_scores'