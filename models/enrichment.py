from f_models import Models, Deploy
import parameters


# instancia a classe
tes = Models()

# treina e gera todos os modelos
if parameters.execute_train_models:
    tes.train_models(parameters)

# instancia o deploy
back = Deploy()

# gera o arquivo de backtest
if parameters.execute_backtest: 
    back.historical_outcome(tes, parameters)
# print((parameters.dados_indicators).tail())


# gera o resultado de hj
back.daily_outcome(tes, parameters,'')
back.daily_outcome(tes, parameters,'2024-01-10')
