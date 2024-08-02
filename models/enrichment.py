from f_models import Models, Deploy
import parameters

# parameters = 

tes = Models()

# tes.train_models(parameters)

back = Deploy()

# back.backtest(tes, parameters)
print(parameters.dados_indicators)

back.daily_outcome(tes, parameters,'')