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


    def save_model(self, classifier, root_path, name_model:str, version_id:str):
        # Lib to save the model in a compressed way

        root_path_version = root_path + '\\' + version_id

        if not os.path.exists(root_path_version):
            # Cria a pasta
            print('pasta nao existe')
            os.makedirs(root_path_version)

        # Save the model that has been trained
        joblib.dump(classifier, root_path_version + '\\' + name_model + '.joblib')

        print(f'Modelo salvo em {root_path_version} com o nome de {name_model}.joblib')


    def load_model(self, name_model:str):
        # Lib to save the model in a compressed way
        import joblib

        # Load the trained model
        clf_loaded = joblib.load(name_model + '.joblib')

        return clf_loaded


    def create_model(self, model_cls, model_name, version_model, root_path, target_eval, X_train, y_train, X_test, y_test):
            
        clf = model_cls.fit(X_train, y_train)

        eval_model_tuple = self.eval_model(clf, X_test, y_test)

        name_model = model_name + '_' + version_model + '_' + target_eval

        var_proba_name = self.build_var_name(name_model, '_pb_')

        self.build_log_model(name_model, var_proba_name, target_eval, eval_model_tuple, version_model)

        self.save_model(clf, root_path, name_model, version_model)

    # Data Flow

    # input_path = r'D:\Github\Forked\crypto_technical_analysis\files\crypto_data_prep_models.parquet'
    # root_path = rf'D:\Github\Forked\crypto_technical_analysis\models\trained'

    def iterate_models(self, parameters):

        # # input_folder = '../scripts/utils/files/'
        # input_folder = 'files/'
        # input_file = 'crypto_data_prep_models.parquet'
        # input_path = Path(input_folder) / input_file
        # dados = pd.read_parquet(input_path)

        # # dados = dados[dados['Symbol'] == 'SOL-USD']

        # # input_folder = '../scripts/utils/files/'
        # save_in_folder = 'models/trained/'
        # root_path = str(Path(save_in_folder))


        # target_list_bol =   [
        #     # booleans positive
        #     'bl_target_10P_7d','bl_target_15P_7d','bl_target_20P_7d','bl_target_25P_7d',
        #     'bl_target_10P_15d','bl_target_15P_15d','bl_target_20P_15d','bl_target_25P_15d', 
        #     'bl_target_10P_30d','bl_target_15P_30d','bl_target_20P_30d','bl_target_25P_30d',
        #     # booleans negative
        #     'bl_target_10N_7d','bl_target_15N_7d','bl_target_20N_7d','bl_target_25N_7d',
        #     'bl_target_10N_15d','bl_target_15N_15d','bl_target_20N_15d','bl_target_25N_15d', 
        #     'bl_target_10N_30d','bl_target_15N_30d','bl_target_20N_30d','bl_target_25N_30d' 
        # ]

        # target_list_val =   [
        #     # real percentual
        #     'target_7d','target_15d','target_30d'
        # ]

        # ## vai gerar a pasta em models/trained/versao se a versa já estiver criada, sobrescreve, se não, cria a pasta
        # version_model = 'v1.4'

        # remove_target_list = target_list_bol + target_list_val

        dados_x = self.data_clean(parameters.dados, parameters.remove_target_list, 'X')
        dados_y_all = self.data_clean(parameters.dados, parameters.remove_target_list, 'Y')


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
            self.create_model(LogisticRegression(class_weight='balanced',random_state=0,max_iter=1000), 'logistic_regression', parameters.version_model, parameters.root_path, target_eval, X_train_norm, y_train, X_test_norm, y_test)
            self.create_model(SVC(probability=True, kernel='linear', C=0.7, max_iter=1000), 'SVC', parameters.version_model, parameters.root_path, target_eval, X_train_norm, y_train, X_test_norm, y_test)

            # Não necessitam de dados normalizados
            self.create_model(RandomForestClassifier(), 'random_forest', parameters.version_model, parameters.root_path, target_eval, X_train, y_train, X_test, y_test)
            self.create_model(XGBClassifier(), 'XGB', parameters.version_model, parameters.root_path, target_eval, X_train, y_train, X_test, y_test)


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

        # # Adicionar as colunas faltantes no conjunto de validação, preenchidas com zeros
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
        # print(proba_target)
        proba_dataset = dummies_before_norm[[]] # pegando apenas os índices do dataset de input (que já contém os dados de retorno)

        proba_dataset[col_name_output] = proba_target

        # proba_crypto_date = dataset_ref[['Symbol', 'Date', 'Close']]

        build_dataset_proba = pd.merge(dataset_ref, proba_dataset, left_index=True, right_index=True)
        # print(build_dataset_proba)
        return build_dataset_proba
        

    def build_compound_proba(self, dados, accuracy_models_dict, score_var_end):

        score_var = 'score_' + score_var_end

        dados[score_var] = 0

        # selecionando os target que possuem o mesmo timeframe e a mesma tendência (Positivo ou Negativo)
        accuracy_models_dict_var = {k: v for k, v in accuracy_models_dict.items() if k.endswith(score_var_end)}

        # print(accuracy_models_dict_var)
        for col in dados.columns:
            # Feito apenas para as colunas de probabilide (que possuem _pb_)
            try: 
                if (col.split('_pb_')[1] is not None) and col.endswith(score_var_end):
                    # print(dados)
                    # print(col.endswith(score_var_end))
                    # Coletando a acurácia do modelo
                    score_model = accuracy_models_dict_var[col]
                    # Normalizar os pesos para que somem 1
                    pondered_score = score_model / sum(accuracy_models_dict_var.values())
                    # print(score_model)


                    # print(sum(accuracy_models_dict_var.values()))
                    # Probabilidade ponderada entre targets
                    pondered_proba = dados[col] * pondered_score
                    dados[score_var] = dados[score_var] + pondered_proba
            except:
                pass
        # print(dados)
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

    
    def build_crypto_scores(self):
            
        # input_folder = '../scripts/utils/files/'
        input_folder = 'files/'

        input_file = 'crypto_data_prep_models.parquet'
        input_path = Path(input_folder) / input_file
        dados = pd.read_parquet(input_path)

        # dados = dados[dados['Symbol'] == 'SOL-USD']



        # input_folder = '../models/'
        input_folder = 'models/'

        input_file2 = 'accuracy/log_models.csv'
        log_models_path = Path(input_folder) / input_file2

        # Definir o diretório que você quer listar os arquivos
        version_id = 'v1.5'
        # input_folder = '../models/'
        input_folder = 'models/'

        # directory = f'trained/{version_id}/'
        directory = f'models/trained/{version_id}/'


        log_models = pd.read_csv(log_models_path)


        # Listar todos os itens no diretório e filtrar apenas os arquivos
        models = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        # Constants
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

        remove_target_list = target_list_bol + target_list_val

        removing_cols = ['Date', 'Symbol', 'Dividends', 'Stock Splits']

        # Acuária dos modelos
        accuracy_models_select = accuracy_models(log_models, version_id)
        # print(accuracy_models_select)

        def main(dados, choosen_data_input = '', backtest = 0):
            # Colocar '' caso deseje a data mais recente presente na base. 
            # Caso colocar em uma data em específico seguir o exemplo: 2024-07-12 
            dataset_ref = eval_data(dados, choosen_data_input)

            dummies_input = build_dummies(dataset_ref, remove_target_list, removing_cols)

            dados_x_all = data_clean(dados, remove_target_list, 'X')
            dados_x_all_dummies = pd.get_dummies(dados_x_all)
            
            
            # dummies_input = norm_scale(dummies_input)
            # dados_x_all_dummies = norm_scale(dados_x_all_dummies)
            # print(dados_x_all_dummies)
            padronized_dummies = padronize_dummies(dummies_input, dados_x_all_dummies)
            # print(padronized_dummies)
            padronized_dummies_norm = norm_scale(padronized_dummies)
            # print(padronized_dummies_norm)

            # padronized_dummies_norm = np.array(padronized_dummies_norm.values)

            compiled_dataset = dataset_ref[['Symbol', 'Date', 'Close']]

            # Iteração para cada modelo na pasta de modelos
            for model in models:
                # print(model)
                clf = joblib.load(directory + model)

                var_proba_name = build_var_name(model, '_pb_')
                compiled_dataset = add_proba_target(clf, padronized_dummies_norm, padronized_dummies, compiled_dataset, var_proba_name)

                # print(compiled_dataset)
            # Mede a probabilidade de todos os targets / modelos, e compoe apenas uma métrica
            compound_proba = self.build_compound_proba(compiled_dataset, accuracy_models_select, 'P_30d')
            compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'P_15d')
            compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'P_7d')
            compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'N_30d')
            compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'N_15d')
            compound_proba = self.build_compound_proba(compound_proba, accuracy_models_select, 'N_7d')


            if backtest == 0:
                print(compound_proba)

                # Salvar o DataFrame em um arquivo CSV
                compound_proba.to_csv(f'models/results/proba_scores_{str(compound_proba['Date'].max())}.csv', index=True)
                # compound_proba.to_csv(f'../models/results/proba_scores_{str(compound_proba['Date'].max())}.csv', index=True)


                print(f'Arquivo salvo em models/results/proba_scores/{str(compound_proba['Date'].max())}.csv')
            else:

                return compound_proba
            



        if __name__ == "__main__":

            # 1 For backtest and build one file for the historical 
            # 0 For the last available date
            backtest = 0

            if backtest == 1:
                start_date = '2024-01-01'
                # today_date = datetime.today().strftime('%Y-%m-%d')
                last_date = str(dados['Date'].max())

                # Gerar um range de datas
                datas = pd.date_range(start=start_date, end=last_date, freq='D')

                # Converter para formato YYYY-MM-DD
                datas_formatadas = datas.strftime('%Y-%m-%d')
                
                output_dataset = pd.DataFrame()
                
                for data in datas_formatadas:

                    output_dataset_date = main(dados, str(data), 1)

                    output_dataset = pd.concat([output_dataset, output_dataset_date])
                    
                    # print(output_dataset)
                # Salvar o DataFrame em um arquivo CSV
                # output_dataset.to_csv(f'../models/results/compound_historical.csv', index=True)
                output_dataset.to_csv(f'models/results/compound_historical.csv', index=True)


                print(f'Arquivo salvo em models/results/compound_historical.csv')
            
            else:
                print(main(dados, '', 0))
