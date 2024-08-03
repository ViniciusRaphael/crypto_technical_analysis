from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler # pip install imblearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from datetime import datetime
import joblib
import os
import pyarrow as pa
import pyarrow.parquet as pq
import warnings

warnings.filterwarnings("ignore")


class DataPrep():

    def __init__(self) -> None:
        pass

    
    def clean_date(self, dados_date):

        dados_date['Date'] = pd.to_datetime(dados_date['Date'])
        dados_date['Date'] = dados_date['Date'].dt.strftime('%Y-%m-%d')

        return dados_date
    

    def save_dataframe_to_parquet(self, dataframe, file_path):
        """
        Save a Pandas DataFrame as a Parquet file.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame to be saved as a Parquet file.
        - file_path (str): Path where the Parquet file will be saved.
        """
        table = pa.Table.from_pandas(dataframe)
        pq.write_table(table=table, where=file_path, compression='snappy')


    def get_active_symbols(self, historical_data):
        max_date = str(historical_data['Date'].max())

        active = (historical_data[historical_data['Date'] == max_date])
        active = active['Symbol']

        list_unique_active = list(set(active))

        return list_unique_active
    


    def build_data_prep_models_file(self, parameters):
        
        cleaned_date = self.clean_date(parameters.dados_indicators)

        active_symbols = self.get_active_symbols(parameters. dados_indicators)

        # Filter to clean data
        filtered_data = cleaned_date[cleaned_date['Symbol'].isin(active_symbols)]

        dados_prep = filtered_data[(filtered_data['Close'] != 0) & (filtered_data['Volume'] > parameters.min_volume_prep_models)]

        if parameters.clean_targets_prep_models == True:
            dados_prep = dados_prep[(dados_prep['target_7d'] < 3) & (dados_prep['target_7d'] > - 0.9) & (dados_prep['target_15d'] < 3) & (dados_prep['target_15d'] > - 0.9) & (dados_prep['target_30d'] < 3) & (dados_prep['target_30d'] > - 0.9)]
        
        self.save_dataframe_to_parquet(dados_prep, parameters.input_path_prep_models)    

        print(f"Parquet file with indicators prep models saved to {parameters.input_path_prep_models} com {len(dados_prep)} linhas")

        return dados_prep



class Models():

    def __init__(self) -> None:
        pass


    def data_clean(self, dados:pd.DataFrame, target_list:list, data_return:str, removing_cols:list = ['Date', 'Symbol', 'Dividends', 'Stock Splits']):
        # Removing NA
        dados_treat = dados.dropna()
        
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

        return confusion_matrix_cal, auc, score


    def build_var_name(self, model_name, prefix):

        string_name = model_name.split('.joblib')[0]

        splited_list = string_name.split('_')
        
        build_name = splited_list[0][0] + splited_list[1][0]  + prefix + splited_list[-2] + '_' + splited_list[-1]
        
        return build_name


    def build_log_model(self, name_file, name_model, target_col, eval_model_tuple, version, dest_path = 'models/accuracy/log_models.csv'): 

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

        root_path_version = parameters.trained_models_path + '\\' + parameters.version_model

        if not os.path.exists(root_path_version):
            # Cria a pasta
            os.makedirs(root_path_version)

        # Save the model that has been trained
        joblib.dump(classifier, root_path_version + '\\' + name_model + '.joblib')

        print(f'Modelo salvo em {root_path_version} com o nome de {name_model}.joblib')


    def create_model(self, parameters, model_cls, model_name, target_eval, X_train, y_train, X_test, y_test):
            
        clf = model_cls.fit(X_train, y_train)

        eval_model_tuple = self.eval_model(clf, X_test, y_test)
    
        name_model = model_name + '_' + parameters.version_model + '_' + target_eval

        var_proba_name = self.build_var_name(name_model, '_pb_')

        self.build_log_model(name_model, var_proba_name, target_eval, eval_model_tuple, parameters.version_model)
        self.save_model(parameters, clf, name_model)


    def train_models(self, parameters):

        dados_x = self.data_clean(parameters.dados_prep_models, parameters.remove_target_list, 'X', parameters.removing_cols)
        dados_y_all = self.data_clean(parameters.dados_prep_models, parameters.remove_target_list, 'Y', parameters.removing_cols)

        for target_eval in parameters.target_list_bol:

            # escolhendo o target
            dados_y = self.get_target(dados_y_all, target_eval)

            # separando em test e treino
            X_train, X_test, y_train, y_test = self.split_data(dados_x, dados_y, 0.3)

            # balanceando classes (Caso necessário)
            X_train, y_train = self.balance_sample(X_train, y_train, 1)

            # Normalizando datasets de treino e teste
            X_train_norm = self.norm_scale(X_train)
            X_test_norm = self.norm_scale(X_test)


            # Utilizam dados normalizados
            self.create_model(parameters, LogisticRegression(class_weight='balanced',random_state=0,max_iter=1000), 'logistic_regression', target_eval, X_train_norm, y_train, X_test_norm, y_test)
            self.create_model(parameters, SVC(probability=True, kernel='linear', C=0.7, max_iter=1000), 'SVC', target_eval, X_train_norm, y_train, X_test_norm, y_test)

            # Não necessitam de dados normalizados
            self.create_model(parameters, RandomForestClassifier(), 'random_forest', target_eval, X_train, y_train, X_test, y_test)
            self.create_model(parameters, XGBClassifier(), 'XGB', target_eval, X_train, y_train, X_test, y_test)


class Deploy():

    def __init__(self) -> None:
        pass


    def eval_data(self, dados, date_eval = None):

        if date_eval is None or date_eval == '':
            choosen_date = dados['Date'].max()
        else:
            choosen_date = date_eval
            
        filtered_data = dados[dados['Date'] == str(choosen_date)]

        return filtered_data


    def build_dummies(self, filtered_data, target_list, remove_cols):
    
        dados_x = filtered_data.drop(filtered_data[target_list], axis=1)
        dados_x = dados_x.drop(dados_x[remove_cols], axis=1)

        dummies_build = pd.get_dummies(dados_x)

        dummies_prep = dummies_build.dropna()

        return dummies_prep


    def padronize_dummies(self, dummies_input, dummies_ref):
        # Encontrar colunas que estão no treinamento mas não na validação
        missing_cols = set(dummies_ref.columns) - set(dummies_input.columns)

        # Adicionar as colunas faltantes no conjunto de validação, preenchidas com zeros
        for col in missing_cols:
            dummies_input[col] = False

        # Reordenar as colunas no conjunto de validação para corresponder ao conjunto de treinamento
        valid_dummies = dummies_input[dummies_ref.columns]

        return valid_dummies
    

    def add_proba_target(self, classifier, dummies_input, dummies_before_norm, dataset_ref, col_name_output):

        #Fazendo a previsão das probabilidades
        proba = classifier.predict_proba(dummies_input)

        # Probabilidade de ser o target:
        proba_target = proba[:,1] # array

        proba_dataset = dummies_before_norm[[]] # pegando apenas os índices do dataset de input (que já contém os dados de retorno)

        proba_dataset[col_name_output] = proba_target
        
        # print(proba_dataset)
        build_dataset_proba = pd.merge(dataset_ref, proba_dataset, left_index=True, right_index=True)

        return build_dataset_proba
        

    def build_compound_proba(self, dados, accuracy_models_dict, score_var_end):

        score_var = 'score_' + score_var_end
        dados[score_var] = 0

        # selecionando os target que possuem o mesmo timeframe e a mesma tendência (Positivo ou Negativo)
        accuracy_models_dict_var = {k: v for k, v in accuracy_models_dict.items() if k.endswith(score_var_end)}

        for col in dados.columns:
            # Feito apenas para as colunas de probabilide (que possuem _pb_)
            try: 
                if (col.split('_pb_')[1] is not None) and col.endswith(score_var_end):
                    # Coletando a acurácia do modelo
                    score_model = accuracy_models_dict_var[col]
                    # Normalizar os pesos para que somem 1
                    pondered_score = score_model / sum(accuracy_models_dict_var.values())
                    # print(score_model)

                    # Probabilidade ponderada entre targets
                    pondered_proba = dados[col] * pondered_score
                    # print(dados[col])
                    dados[score_var] = dados[score_var] + pondered_proba
            except:
                pass

        return dados.sort_values(by=score_var, ascending=False)


    def accuracy_models(self, log_models, version):

        accuracy_dict = {}

        log_models_select = log_models[log_models['version'] == version]
        
        # só vai ter o problema se o modelo estiver rodando as meia noite, porque ai vai ter dois dias
        log_models_select = log_models_select[log_models_select['date_add'] == log_models_select['date_add'].max()]

        for idx, row in log_models_select.iterrows():
            name_model_select = row['name_model']
            accuracy_model_select = row['accuracy']

            accuracy_dict[name_model_select] = accuracy_model_select
        
        return accuracy_dict

    
    def build_crypto_scores(self, cls_Models, parameters, choosen_data_input = '', backtest = False):
        
        dados_input_select = parameters.dados_prep_models if backtest else parameters.dados_indicators

        # Listar todos os itens no diretório e filtrar apenas os arquivos
        models = [f for f in os.listdir(parameters.directory_models) if os.path.isfile(os.path.join(parameters.directory_models, f))]

        # Acuária dos modelos
        # log_models = pd.read_csv(parameters.log_models_path)

        accuracy_models_select = self.accuracy_models(parameters.log_models, parameters.version_model)

        # accuracy_models_select = pd.read_csv(parameters.log_models_path)


        # def main(dados, choosen_data_input = '', backtest = 0):
        # Colocar '' caso deseje a data mais recente presente na base. 
        # Caso colocar em uma data em específico seguir o exemplo: 2024-07-12 
        dataset_ref = self.eval_data(dados_input_select, choosen_data_input)

        dummies_input = self.build_dummies(dataset_ref, parameters.remove_target_list, parameters.removing_cols)

        #aq
        dados_x_all = cls_Models.data_clean(dados_input_select, parameters.remove_target_list, 'X', parameters.removing_cols)
        dados_x_all_dummies = pd.get_dummies(dados_x_all)
        
        padronized_dummies = self.padronize_dummies(dummies_input, dados_x_all_dummies)
        padronized_dummies_norm = cls_Models.norm_scale(padronized_dummies)

        compiled_dataset = dataset_ref[['Symbol', 'Date', 'Close']]


        # Iteração para cada modelo na pasta de modelos
        for model in models:
            clf = joblib.load(parameters.directory_models + model)

            var_proba_name = cls_Models.build_var_name(model, '_pb_')
            compiled_dataset = self.add_proba_target(clf, padronized_dummies_norm, padronized_dummies, compiled_dataset, var_proba_name)

        # Mede a probabilidade de todos os targets / modelos, e compoe apenas uma métrica
        compound_proba = self.build_compound_proba(compiled_dataset, accuracy_models_select, 'P_30d')
        compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'P_15d')
        compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'P_7d')
        compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'N_30d')
        compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'N_15d')
        compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'N_7d')

        return compound_proba
        


    def historical_outcome(self, cls_Models, parameters):

        start_date = parameters.start_date_backtest
        last_date = str(parameters.dados_prep_models['Date'].max())

        # Gerar um range de datas
        datas = pd.date_range(start=start_date, end=last_date, freq='D')

        # Converter para formato YYYY-MM-DD
        datas_formatadas = datas.strftime('%Y-%m-%d')
        
        backtest_dataset = pd.DataFrame()
        
        for data in datas_formatadas:

            print(f'Backtesting dia {data}')

            backtest_dataset_date = self.build_crypto_scores(cls_Models, parameters, str(data), True)
            backtest_dataset = pd.concat([backtest_dataset, backtest_dataset_date])
            
        # Salvar o DataFrame em um arquivo CSV
        backtest_dataset.to_csv(parameters.backtest_path, index=True)

        print(f'Arquivo salvo em {parameters.backtest_path}')
    
        return backtest_dataset
        
    
    def daily_outcome(self, cls_Models, parameters, choosen_date):
        
        daily_outcome = self.build_crypto_scores(cls_Models, parameters, choosen_date, False)
        print(daily_outcome)

        # Salvar o DataFrame em um arquivo CSV
        file_name_outcome = f"{parameters.daily_outcome_path}_{str(daily_outcome['Date'].max())}.csv"
        daily_outcome.to_csv(file_name_outcome, index=True)

        print(f'Arquivo salvo em {file_name_outcome}')

        return daily_outcome


