import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler # pip install imblearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from datetime import datetime
import joblib
import os
import warnings


warnings.filterwarnings("ignore")

class Models():

    def __init__(self) -> None:
        pass


    def data_clean(self, dados:pd.DataFrame, target_list:list, data_return:str, removing_cols:list = ['Date', 'Symbol', 'Dividends', 'Stock Splits']):
        # Definir o limite de 50% para valores inválidos (NaN ou infinitos)
        limite = len(dados) * 0.8

        dados_treat = dados.dropna(thresh=limite, axis=1) # removing cols with all values = NaN

        dados_treat = dados_treat.dropna() # Removing rows with NaN

        # Substituindo valores infinitos por NaN
        dados_treat.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Removendo linhas com valores NaN
        dados_treat.dropna(inplace=True)

        # Define the target in a list of target (for futher iteration)
        dados_y = dados_treat[target_list]

        # Removing target from base to avoid data leakage
        dados_x = dados_treat.drop(dados_treat[target_list], axis=1)
        dados_x = dados_x.drop(dados_x[removing_cols], axis=1)

        if data_return == 'Y':
            return dados_y
        else:
            return dados_x



    def get_target(self, dados_y:pd.DataFrame, col_target:str):
        return dados_y[col_target]


    def split_data(self, dados_x:pd.DataFrame, dados_y:pd.DataFrame, test_size:float=0.3):

        # Getting dummies values. This way we can use categorical columns to train the models
        dummies = pd.get_dummies(dados_x)

        # Transform the data in numpy arrays
        X = np.array(dummies.values)
        y = np.array(dados_y.values)

        # This function returns X_train, X_test, y_train, y_test, in this order.
        # See below an example of the return 
        # X_train, X_test, y_train, y_test = split_data(dados_x, dados_y, 0.3)

        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=45)


    def norm_scale(self, X_norm_scale):

        # normalizando e padronizando os dados
        # MinMaxScaler é usado para normalizar as variáveis, colocando em uma mesma escala,
        # e StandardScaler é usado para padronizar, fazendo com que a média seja 0 e o desvio padrão seja 1
        # Padronizando
        scaler = StandardScaler()
        scaler.fit(X_norm_scale)
        standardized_data = scaler.transform(X_norm_scale)

        # normalizando
        scaler = MinMaxScaler()
        scaler.fit(standardized_data)
        normalized_data = scaler.transform(standardized_data)
        
        return normalized_data

    # Balanceando as classes 
    def balance_sample(self, X_train, y_train, type):

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
        

    def eval_model(self, classifier, X_test, y_test):

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

        # Calcular o recall (sensibilidade)
        print('Sensibilidade:', confusion_matrix_cal[1,1] / (confusion_matrix_cal[1,1] + confusion_matrix_cal[1,0]))
        # Calcular a precisão
        print('Precisão:', confusion_matrix_cal[1,1] / (confusion_matrix_cal[1,1] + confusion_matrix_cal[0,1]))

        return confusion_matrix_cal, auc, score


    def build_var_name(self, model_name, prefix):

        string_name = model_name.split('.joblib')[0]

        splited_list = string_name.split('_')
        
        build_name = splited_list[0][0] + splited_list[1][0]  + prefix + splited_list[-2] + '_' + splited_list[-1]
        
        return build_name


    def build_log_model(self, name_file, name_model, target_col, eval_model_tuple, version, dest_path): 

        matrix, auc, score_cal = eval_model_tuple
        
        # Dados
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
        df.to_csv(dest_path, mode='a', index=False, header=False)

        return df
    

    def save_model(self, parameters, classifier, name_model:str):
        # Lib to save the model in a compressed way

        root_path_version = parameters.path_models

        if not os.path.exists(root_path_version):
            # Cria a pasta
            os.makedirs(root_path_version)

        # Save the model that has been trained
        joblib.dump(classifier, root_path_version/f'{name_model}.joblib', compress = 5)

        print(f'Modelo salvo em {root_path_version} com o nome de {name_model}.joblib')


    def create_model(self, parameters, model_cls, model_name, target_eval, X_train, y_train, X_test, y_test):
            
        clf = model_cls.fit(X_train, y_train)

        eval_model_tuple = self.eval_model(clf, X_test, y_test)
    
        name_model = model_name + '_' + parameters.version_model + '_' + target_eval

        var_proba_name = self.build_var_name(name_model, '_pb_')

        self.build_log_model(name_model, var_proba_name, target_eval, eval_model_tuple, parameters.version_model, parameters.file_log_models)
        self.save_model(parameters, clf, name_model)


    def train_models(self, parameters):

        dados_prep_models = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_prep_models)

        dados_prep_models = parameters.cls_FileHandling.get_selected_symbols(dados_prep_models, parameters)

        # Access dict with models configs
        _dict_config_train = parameters.cls_FileHandling.get_constants_dict(parameters, parameters.cls_Constants._get_configs_train())

        dados_x = self.data_clean(dados_prep_models, parameters._remove_target_list, 'X', _dict_config_train['removing_cols_for_train'])
        dados_y_all = self.data_clean(dados_prep_models, parameters._remove_target_list, 'Y', _dict_config_train['removing_cols_for_train'])

        print(len(dados_x.columns))
        # limpar arquivo de log e inserindo o cabeçalho
        with open(parameters.file_log_models, 'w') as arquivo:
            arquivo.write('name_file,name_model,target,version,date_add,true_negative,false_positive,false_negative,true_positive,accuracy,precision,recall,auc_roc,f1_score' + '\n')

        # Counter models     
        c_trained_target = 1

        for target_eval in parameters.target_list_bol_select:
            
            print(f'Train model target: {c_trained_target}/{len(parameters.target_list_bol_select)}')

            # escolhendo o target
            dados_y = self.get_target(dados_y_all, target_eval)

            # separando em test e treino
            X_train, X_test, y_train, y_test = self.split_data(dados_x, dados_y, 0.3)

            # balanceando classes (Caso necessário)
            X_train, y_train = self.balance_sample(X_train, y_train, 1)

            # Normalizando datasets de treino e teste
            X_train_norm = self.norm_scale(X_train)
            X_test_norm = self.norm_scale(X_test)

            # Access the dict in constants
            _dict_classifiers = parameters.cls_FileHandling.get_constants_dict(parameters, parameters.cls_Constants._get_classifiers())

            # Utilizam dados normalizados
            self.create_model(parameters, _dict_classifiers['lr'], 'logistic_regression', target_eval, X_train_norm, y_train, X_test_norm, y_test)
            # self.create_model(parameters, _dict_classifiers['Sc'], 'SVC', target_eval, X_train_norm, y_train, X_test_norm, y_test)

            # Não necessitam de dados normalizados
            self.create_model(parameters, _dict_classifiers['rf'], 'random_forest', target_eval, X_train, y_train, X_test, y_test)
            self.create_model(parameters, _dict_classifiers['Xv'], 'XGB', target_eval, X_train, y_train, X_test, y_test)

            c_trained_target += 1

