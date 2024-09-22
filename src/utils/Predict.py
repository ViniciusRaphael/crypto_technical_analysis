import pandas as pd
import joblib
import os
import warnings



warnings.filterwarnings("ignore")

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

        # Cols with all the values Null
        null_cols = filtered_data.columns[filtered_data.isnull().all()]
        null_cols_withou_targets = [x for x in null_cols if x not in target_list]

        print(f'Removed Null cols: {null_cols_withou_targets}')

        # dados_x_dropped = dados_x_dropped.dropna() # Removing rows with NaN
        dados_x_dropped = filtered_data.drop(columns=null_cols_withou_targets, axis=1)
        
        dados_x = dados_x_dropped.drop(dados_x_dropped[target_list], axis=1)
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


    def accuracy_models(self, log_models, metric):

        accuracy_dict = {}

        for idx, row in log_models.iterrows():
            name_model_select = row['name_model']
            accuracy_model_select = row[metric] #antes era accuracy

            accuracy_dict[name_model_select] = accuracy_model_select
        
        return accuracy_dict
    

    # def accuracy_models(self, log_models, version):

    #     accuracy_dict = {}

    #     log_models_select = log_models[log_models['version'] == version]
        
    #     # só vai ter o problema se o modelo estiver rodando as meia noite, porque ai vai ter dois dias
    #     log_models_select = log_models_select[log_models_select['date_add'] == log_models_select['date_add'].max()]

    #     for idx, row in log_models_select.iterrows():
    #         name_model_select = row['name_model']
    #         accuracy_model_select = row['accuracy']

    #         accuracy_dict[name_model_select] = accuracy_model_select
        
    #     return accuracy_dict

    def choose_best_models(self, parameters):

        log_models = pd.read_csv(parameters.file_log_models)

        if parameters.min_threshold_models > 0:
            log_models = log_models[log_models[parameters.score_metric] >= parameters.min_threshold_models]
        
        if parameters.num_select_models > 0:
            log_models = log_models.sort_values(by=parameters.score_metric, ascending=False).head(parameters.num_select_models)

        return log_models

    
    def build_crypto_scores(self, cls_Models, parameters, choosen_data_input = '', backtest = False):
        
        dados_prep_models = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_prep_models)

        dados_w_indicators = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_w_indicators)

        dados_input_select = dados_prep_models if backtest else dados_w_indicators

        dados_input_select = parameters.cls_FileHandling.get_selected_symbols(dados_input_select, parameters) if parameters.execute_filtered else dados_input_select

        # Listar todos os itens no diretório e filtrar apenas os arquivos
        # models = [f for f in os.listdir(parameters.path_models) if os.path.isfile(os.path.join(parameters.path_models, f))]

        # Acuária dos modelos
        # log_models = pd.read_csv(parameters.file_log_models)
        log_models = self.choose_best_models(parameters)

        models = list(set(log_models['name_file']))

        accuracy_models_select = self.accuracy_models(log_models, parameters.score_metric)

        # Vai escolher uma data em específico, '' para a mais recente
        dataset_ref = self.eval_data(dados_input_select, choosen_data_input)
        

        # Access dict with models configs
        _dict_config_train = parameters.cls_FileHandling.get_constants_dict(parameters, parameters.cls_Constants._get_configs_train())

        dummies_input = self.build_dummies(dataset_ref, parameters._remove_target_list, _dict_config_train['removing_cols_for_train'])

        # padronize input parameters from test models x predict model 
        dados_x_all = cls_Models.data_clean(dados_input_select, parameters._remove_target_list, 'X', _dict_config_train['removing_cols_for_train'])
        dados_x_all_dummies = pd.get_dummies(dados_x_all)

        padronized_dummies = self.padronize_dummies(dummies_input, dados_x_all_dummies)

        padronized_dummies_norm = cls_Models.norm_scale(padronized_dummies)

        compiled_dataset = dataset_ref[['Symbol', 'Date', 'Close']]

        # Iteração para cada modelo na pasta de modelos
        for model in models:
            
            clf = joblib.load(str(parameters.path_models / model) + '.joblib')

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

        dados_prep_models = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_prep_models)

        last_date = str(dados_prep_models['Date'].max())

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
        backtest_dataset.to_csv(parameters.file_backtest, index=True)

        print(f'Arquivo salvo em {parameters.file_backtest}')
    
        return backtest_dataset
        
    
    def daily_outcome(self, cls_Models, parameters, choosen_date):
        
        print(f'Predicting selected date')

        daily_outcome = self.build_crypto_scores(cls_Models, parameters, choosen_date, False)
        # print(daily_outcome)

        # Salvar o DataFrame em um arquivo CSV
        if not os.path.exists(parameters.path_daily_outcome):
            # Cria a pasta
            os.makedirs(parameters.path_daily_outcome)

        file_name_outcome = f"{parameters.path_daily_outcome}/proba_scores_{str(daily_outcome['Date'].max())}.csv"

        # Transform the wide dataset into long
        if parameters.melt_daily_predict == True:
            daily_outcome = daily_outcome.melt(id_vars=['Symbol', 'Date', 'Close'], var_name='Models', value_name='Probability')
            # Join with simple backtest to rescue the value for backtesting with all currencies
            try: 
                daily_output_filename = f'{parameters.path_model_backtest}/_simple_backtest_{parameters.version_model}_{parameters.min_threshold_signals}_.csv'
                backtest_file_selected = pd.read_csv(daily_output_filename, sep=';')
                daily_outcome = pd.merge(daily_outcome, backtest_file_selected[['Symbol', 'model', 'number_entries', 'percent_correct_entries', 'simulate_variation', 'reached_target']],
                                                                            how='left', left_on=['Symbol', 'Models'], right_on=['Symbol', 'model'])

                daily_outcome = daily_outcome.sort_values(by=['reached_target', 'Probability', 'simulate_variation'], ascending=[False, False, False])
            except:
                print('Warning: No backtest file associated with the predicted result. Returning dataset without the backtest enrichment')
                pass
            
            print(daily_outcome.head(20))

        daily_outcome.to_csv(file_name_outcome, index=True, sep=';', decimal=',')

        print(f'Arquivo salvo em {file_name_outcome}')

        return daily_outcome


