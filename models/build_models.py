from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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


def get_target(dados_y:pd.DataFrame, col_target:str):
    return dados_y[col_target]


def split_data(dados_x:pd.DataFrame, dados_y:pd.DataFrame, test_size:float=0.3):

    # Getting dummies values. This way we can use categorical columns to train the models
    dummies = pd.get_dummies(dados_x)

    # Transform the data in numpy arrays
    X = np.array(dummies.values)
    y = np.array(dados_y.values)

    # This function returns X_train, X_test, y_train, y_test, in this order.
    # See below an example of the return 
    # X_train, X_test, y_train, y_test = split_data(dados_x, dados_y, 0.3)

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=45)


# def norm_scale(X_norm_scale):

#     # normalizando e padronizando os dados
#     # MinMaxScaler é usado para normalizar as variáveis, e StandardScaler é usado para padronizar
#     from sklearn.preprocessing import MinMaxScaler, StandardScaler

#     # normalizando
#     scaler = MinMaxScaler()
#     scaler.fit(X_norm_scale)
#     normalized_data = scaler.transform(X_norm_scale)
#     # print(normalized_data)

#     # Padronizando
#     scaler = StandardScaler()
#     scaler.fit(X_norm_scale)
#     standardized_data = scaler.transform(X_norm_scale)
#     # print(standardized_data)

#     # print(standardized_data.shape)
    
#     return standardized_data

def norm_scale(X_norm_scale):

    # normalizando e padronizando os dados
    # MinMaxScaler é usado para normalizar as variáveis, colocando em uma mesma escala,
    # e StandardScaler é usado para padronizar, fazendo com que a média seja 0 e o desvio padrão seja 1
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # Padronizando
    scaler = StandardScaler()
    scaler.fit(X_norm_scale)
    standardized_data = scaler.transform(X_norm_scale)
    # print(standardized_data.shape)

    # normalizando
    scaler = MinMaxScaler()
    scaler.fit(standardized_data)
    normalized_data = scaler.transform(standardized_data)
    # print(normalized_data)
    
    return normalized_data

# Balanceando as classes 
def balance_sample(X_train, y_train, type):

    from imblearn.under_sampling import RandomUnderSampler # pip install imblearn
    from imblearn.over_sampling import SMOTE

    # Reduzir amostra
    if type == 1:
        undersampler = RandomUnderSampler(random_state=42)
        return undersampler.fit_resample(X_train, y_train)

    # Aumentar amostra
    if type == 2:
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train, y_train)
    
    # Sem mudanças
    if type == 0:
        return X_train, y_train   
    

def eval_model(classifier, X_test, y_test):

    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    from sklearn.metrics import roc_auc_score

    # Fazendo a previsão das classes
    y_pred2 = classifier.predict(X_test)

    confusion_matrix_cal = confusion_matrix(y_test,y_pred2)

    # Avaliando o erro
    print('Confusion Matrix')
    print(confusion_matrix_cal)

    #Fazendo a previsão das probabilidades
    proba = classifier.predict_proba(X_test)

    # Probabilidade de ser o target:
    proba_target = proba[:,1] # array

    # Calcular AUC ROC
    auc = roc_auc_score(y_test, proba_target)
    print("AUC ROC:", auc)

    # Avaliando o modelo 
    # score = model.score(X_test, y_test)
    score = metrics.accuracy_score(y_test, y_pred2)

    # Percentagem de acerto
    print('Acurácia:', score)

    return confusion_matrix_cal, auc, score


def build_var_name(model_name, prefix):

    string_name = model_name.split('.joblib')[0]

    splited_list = string_name.split('_')
    
    build_name = splited_list[0][0] + splited_list[1][0]  + prefix + splited_list[-2] + '_' + splited_list[-1]
    
    return build_name


def build_log_model(name_file, name_model, target_col, eval_model_tuple, version): 

    from datetime import datetime

    matrix, auc, score_cal = eval_model_tuple
    
    # Exemplo de dados
    data = {
        'name_file': [name_file],
        'name_model': [name_model],
        'target': [target_col],
        'version': [version],
        'date_add': datetime.today().strftime('%Y-%m-%d'),
        'true_negative': [matrix[0,0]],
        'false_positive': [matrix[0,1]],
        'false_negative': [matrix[1,0]],
        'true_positive':[ matrix[1,1]],
        'accuracy': [score_cal],
        'precision': [matrix[1,1] / (matrix[1,1] + matrix[0,1])],   # Proporção de previsões positivas corretas em relação ao total de previsões positivas.
        'recall': [matrix[1,1] / (matrix[1,1] + matrix[1,0])], #Revocação (Recall) ou Sensibilidade (Sensitivity): Proporção de casos positivos corretamente identificados.
        'auc_roc': [auc]
    }

    df = pd.DataFrame(data)

    df['f1_score'] = 2 * ((df['precision'] * df['recall']) / (df['precision'] + df['recall'])) # F1 Score: Média harmônica da precisão e da revocação, usada para balancear os trade-offs entre essas duas métricas.

    # Apendar o DataFrame em um arquivo CSV de resultado
    df.to_csv(f'models/accuracy/log_models.csv', mode='a', index=False, header=False)

    return df


def save_model(classifier, name_model:str):
    # Lib to save the model in a compressed way
    import joblib

    # Save the model that has been trained
    joblib.dump(classifier, name_model + '.joblib')

    print(f'Modelo savo no diretório atual com o nome de {name_model}.joblib')


def load_model(name_model:str):
    # Lib to save the model in a compressed way
    import joblib

    # Load the trained model
    clf_loaded = joblib.load(name_model + '.joblib')

    return clf_loaded


# Data Flow

input_path = r'D:\Github\Forked\crypto_technical_analysis\files\crypto_data_prep_models.parquet'

dados = pd.read_parquet(input_path)

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

version_model = 'v1.4'

remove_target_list = target_list_bol + target_list_val

dados_x = data_clean(dados, remove_target_list, 'X')
dados_y_all = data_clean(dados, remove_target_list, 'Y')


for target_eval in target_list_bol:

    # escolhendo o target
    dados_y = get_target(dados_y_all, target_eval)

    # separando em test e treino
    X_train, X_test, y_train, y_test = split_data(dados_x, dados_y, 0.3)

    # balanceando classes (Caso necessário)
    X_train, y_train = balance_sample(X_train, y_train, 2)

    # Normalizando datasets de treino e teste
    X_train = norm_scale(X_train)
    X_test = norm_scale(X_test)

    # Criando o modelo
    model = LogisticRegression(class_weight='balanced',random_state=0,max_iter=1000)

    # Treinando o modelo
    model.fit(X_train, y_train)

    clf = LogisticRegression(random_state=45,max_iter=1000).fit(X_train, y_train)

    eval_model_tuple = eval_model(clf, X_test, y_test)

    name_model = 'logistic_regression_model_' + version_model + '_' + target_eval

    var_proba_name = build_var_name(name_model, '_pb_')

    build_log_model(name_model, var_proba_name, target_eval, eval_model_tuple, version_model)

    save_model(clf, name_model)