import parameters

# instancia as classes
cls_File = parameters.cls_FileHandling
cls_Models = parameters.cls_Models
cls_Deploy = parameters.cls_Predict
cls_DataPrep = parameters.cls_DataPrep


if parameters.execute_data_ingestion:
    parameters.cls_Ingestion.build_historical_data(cls_File, parameters)

if parameters.execute_data_indicators:
    parameters.cls_Transform.build_crypto_indicators(cls_File, parameters)

if parameters.execute_data_prep_models:
    parameters.cls_DataPrep.build_data_prep_models_file(cls_File, parameters)

if parameters.execute_train_models:
    parameters.cls_Models.train_models(parameters)

if parameters.execute_historical_predict: 
    parameters.cls_Predict.historical_outcome(cls_Models, parameters)

if parameters.execute_daily_predict: 
    parameters.cls_Predict.daily_outcome(cls_Models, parameters, '')

if parameters.execute_signals: 
    parameters.cls_Signals.build_signals(parameters)

if parameters.execute_backtest: 
    parameters.cls_RealBacktest.backtest_models(parameters)

if parameters.execute_backtest_simple: 
    parameters.cls_RealBacktest.all_entries_backtest(parameters)






