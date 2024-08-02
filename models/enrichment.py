from f_models import Models, Deploy
import parameters

# parameters = 

tes = Models()

# tes.train_models(parameters)

back = Deploy()

back.backtest(tes, parameters)