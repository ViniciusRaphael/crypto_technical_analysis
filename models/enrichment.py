from f_models import Models, Deploy
import parameters

# parameters = 

# instancia a classe
tes = Models()

# treina e gera todos os modelos
# tes.train_models(parameters)

# instancia o deploy
back = Deploy()

# gera o arquivo de backtest
# back.backtest(tes, parameters)
# print((parameters.dados_indicators).tail())


# gera o resultado de hj
back.daily_outcome(tes, parameters,'')
back.daily_outcome(tes, parameters,'2024-01-10')
