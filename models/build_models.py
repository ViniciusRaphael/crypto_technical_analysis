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
    removing_cols = ['Date']
    # removing_cols = ['Date', 'Symbol']

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


def norm_scale(X_norm_scale):

    # normalizando e padronizando os dados
    # MinMaxScaler é usado para normalizar as variáveis, e StandardScaler é usado para padronizar
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # normalizando
    scaler = MinMaxScaler()
    scaler.fit(X_norm_scale)
    normalized_data = scaler.transform(X_norm_scale)
    # print(normalized_data)

    # Padronizando
    scaler = StandardScaler()
    scaler.fit(X_norm_scale)
    standardized_data = scaler.transform(X_norm_scale)
    # print(standardized_data)

    # print(standardized_data.shape)
    
    return standardized_data

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

    # Fazendo a previsão das classes
    y_pred2 = classifier.predict(X_test)

    # Avaliando o erro
    print('Confusion Matrix')
    print(confusion_matrix(y_test,y_pred2))

    # Avaliando o modelo 
    # score = model.score(X_test, y_test)
    score = metrics.accuracy_score(y_test, y_pred2)

    # Percentagem de acerto
    print('Acurácia:', score)


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

input_path = r'D:\Github\Forked\crypto_technical_analysis\files\crypto_data_with_indicators.parquet'

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

    eval_model(clf, X_test, y_test)

    save_model(clf, 'logistic_regression_model_' + target_eval)