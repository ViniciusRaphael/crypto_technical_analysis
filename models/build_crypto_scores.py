import os
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)


# Essa é a mesma do build (mudar aqui dps em outro MR)
def data_clean(dados:pd.DataFrame, target_list:list, data_return:str):
    # Removing NA
    dados_treat = dados.dropna()

    # Substituindo valores infinitos por NaN
    dados_treat.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Removendo linhas com valores NaN
    dados_treat.dropna(inplace=True)

    # Removing cols that won't be used in the model
    # removing_cols = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    # removing_cols = ['Date']
    # removing_cols = ['Date', 'Symbol']
    removing_cols = ['Date', 'Symbol', 'Dividends', 'Stock Splits']

    # Define the target in a list of target (for futher iteration)
    dados_y = dados_treat[target_list]

    # Removing target from base to avoid data leakage
    dados_x = dados_treat.drop(dados_treat[target_list], axis=1)
    dados_x = dados_x.drop(dados_x[removing_cols], axis=1)

    if data_return == 'Y':
        return dados_y
    else:
        return dados_x
    



def eval_data(dados, date_eval = None):

    dados['Date'] = pd.to_datetime(dados['Date'])
    dados['Date'] = dados['Date'].dt.strftime('%Y-%m-%d')

    if date_eval is None or date_eval == '':
        choosen_date = dados['Date'].max()
    else:
        choosen_date = date_eval
        
    filtered_data = dados[dados['Date'] == choosen_date]

    return filtered_data


def build_dummies(filtered_data, target_list, remove_cols):
    
    dados_x = filtered_data.drop(filtered_data[target_list], axis=1)
    dados_x = dados_x.drop(dados_x[remove_cols], axis=1)

    dummies_build = pd.get_dummies(dados_x)

    dummies_prep = dummies_build.dropna()

    return dummies_prep


def padronize_dummies(dummies_input, dummies_ref):
    # Encontrar colunas que estão no treinamento mas não na validação
    missing_cols = set(dummies_ref.columns) - set(dummies_input.columns)

    # # Adicionar as colunas faltantes no conjunto de validação, preenchidas com zeros
    for col in missing_cols:
        dummies_input[col] = False

    # Reordenar as colunas no conjunto de validação para corresponder ao conjunto de treinamento
    valid_dummies = dummies_input[dummies_ref.columns]

    return valid_dummies


def add_proba_target(classifier, dummies_input, dataset_ref, col_name_output):

    #Fazendo a previsão das probabilidades
    proba = classifier.predict_proba(dummies_input)

    # Probabilidade de ser o target:
    proba_target = proba[:,1] # array

    proba_dataset = dummies_input[[]] # pegando apenas os índices do dataset de input (que já contém os dados de retorno)

    proba_dataset[col_name_output] = proba_target

    # proba_crypto_date = dataset_ref[['Symbol', 'Date', 'Close']]

    build_dataset_proba = pd.merge(dataset_ref, proba_dataset, left_index=True, right_index=True)
    
    return build_dataset_proba


def build_var_name(model_name, prefix):

    string_name = model_name.split('.')[0]

    splited_list = string_name.split('_')
    
    build_name = splited_list[0][0] + splited_list[1][0]  + prefix + splited_list[-2] + '_' + splited_list[-1]
    
    return build_name


def build_compound_proba(dados, accuracy_models_dict):
    dados['score'] = 0

    for col in dados.columns:
        # Feito apenas para as colunas de probabilide (que possuem _pb_)
        try: 
            if col.split('_pb_')[1] is not None:
                # Coletando a acurácia do modelo
                score_model = accuracy_models_dict[col.split('_pb_')[0] + '_ac_' + col.split('_pb_')[1]]
                # Normalizar os pesos para que somem 1
                pondered_score = score_model / sum(accuracy_models_dict.values())
                # Probabilidade ponderada entre targets
                pondered_proba = dados[col] * pondered_score
                dados['score'] = dados['score'] + pondered_proba
        except:
            pass

    return dados.sort_values(by='score', ascending=False)


input_path = r'D:\Github\Forked\crypto_technical_analysis\files\crypto_data_with_indicators.parquet'

dados = pd.read_parquet(input_path)

# Definir o diretório que você quer listar os arquivos
directory = 'models/trained_v1/'

# Listar todos os itens no diretório e filtrar apenas os arquivos
models = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Constants
target_list_bol =   [
    # boleanos
    'bl_target_10_7d','bl_target_15_7d','bl_target_20_7d','bl_target_25_7d',
    'bl_target_10_15d','bl_target_15_15d','bl_target_20_15d','bl_target_25_15d', 
    'bl_target_10_30d','bl_target_15_30d','bl_target_20_30d','bl_target_25_30d' 
]

target_list_val =   [
    # percentual
    'target_10_7d','target_15_7d','target_20_7d','target_25_7d',
    'target_10_15d','target_15_15d','target_20_15d','target_25_15d', 
    'target_10_30d','target_15_30d','target_20_30d','target_25_30d', 
]

remove_target_list = target_list_bol + target_list_val

removing_cols = ['Date', 'Symbol', 'Dividends', 'Stock Splits']

# Acuária dos modelos
accuracy_models = {
    'lr_ac_10_7d': 0.6048148473284414,
    'lr_ac_15_7d': 0.5965732167662444,
    'lr_ac_20_7d': 0.5683833287718628,
    'lr_ac_25_7d': 0.35522150673118136,
    'lr_ac_10_15d': 0.6551115150451069,
    'lr_ac_15_15d': 0.6507789786781375,
    'lr_ac_20_15d': 0.6411998905247068,
    'lr_ac_25_15d': 0.62837269107828,
    'lr_ac_10_30d': 0.6880883651517421,
    'lr_ac_15_30d': 0.6821911583208968,
    'lr_ac_20_30d': 0.6725914144517715,
    'lr_ac_25_30d': 0.6631775720238987
}

dataset_ref = eval_data(dados, '')

dummies_input = build_dummies(dataset_ref, remove_target_list, removing_cols)

dados_x_all = data_clean(dados, remove_target_list, 'X')
dados_x_all_dummies = pd.get_dummies(dados_x_all)

padronized_dummies = padronize_dummies(dummies_input, dados_x_all_dummies)

compiled_dataset = dataset_ref[['Symbol', 'Date', 'Close']]

# Iteração para cada modelo na pasta de modelos
for model in models:
    clf = joblib.load(directory + model)

    var_proba_name = build_var_name(model, '_pb_')

    compiled_dataset = add_proba_target(clf, padronized_dummies, compiled_dataset, var_proba_name)

# Mede a probabilidade de todos os targets / modelos, e compoe apenas uma métrica
compound_proba = build_compound_proba(compiled_dataset, accuracy_models)

print(compound_proba)

# Salvar o DataFrame em um arquivo CSV
compound_proba.to_csv(f'models/results/proba_scores_{str(compound_proba['Date'].max())}.csv', index=True)
