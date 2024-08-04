from f_models import Models, Deploy, DataPrep, FileHandling, DataIngestion, DataTransform

import parameters

# instancia as classes
cls_File = FileHandling()
cls_Models = Models()
cls_Deploy = Deploy()
cls_DataPrep = DataPrep()

if parameters.execute_data_ingestion:
    DataIngestion().build_historical_data(cls_File, parameters)

if parameters.execute_data_indicators:
    DataTransform().build_crypto_indicators(cls_File, parameters)

if parameters.execute_data_prep_models:
    DataPrep().build_data_prep_models_file(cls_File, parameters)

if parameters.execute_train_models:
    Models().train_models(parameters)

if parameters.execute_backtest: 
    Deploy().historical_outcome(cls_Models, parameters)

if parameters.execute_daily_outcome: 
    Deploy().daily_outcome(cls_Models, parameters, '')






