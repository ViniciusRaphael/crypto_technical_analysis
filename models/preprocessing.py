# Importando as bibliotecas
import numpy as np
import pandas as pd
# from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path


class Preprocessing:
    def create_target(dataset:pd.DataFrame, col_name_input:str, col_name_output:str, target_percent: int, days_ahead: int):

        # Calcular o valor 7 dias à frente
        dataset['value_ahead'] = dataset[col_name_input].shift(-days_ahead)

        # Calcular a variação percentual
        dataset['pct_change'] = (dataset['value_ahead'] - dataset[col_name_input]) / dataset[col_name_input] * 100

        # Criar coluna target que verifica se a variação percentual é maior que 10%
        dataset[col_name_output] = dataset['pct_change'] > target_percent

        # Exibir DataFrame com a coluna target
        return dataset

    def data_clean(dataset:pd.DataFrame, target_list:list, choosen_target:str, data_return:str):
        ##Removendo NA
        dados = dataset.dropna()

        cols_target = target_list 
        # ['target_10_7d','target_15_7d','target_20_7d','target_25_7d',
        #                 'target_10_15d','target_15_15d','target_20_15d','target_25_15d', 
        #                 'target_10_30d','target_15_30d','target_20_30d','target_25_30d']

        cols_target.insert(0,'Symbol')
        cols_target.insert(0,'Date')

        # Definindo o target, fazer uma iteração aqui
        dados_y = dados[choosen_target]

        # Retirando todos os target da base para evitar data leakage
        dados_x = dados.drop(dados[cols_target], axis=1)

        if data_return == 'Y':
            return dados_y
        elif data_return == 'X':
            return dados_x
        else:
            return None