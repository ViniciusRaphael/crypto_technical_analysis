from pathlib import Path


class DataPrep():

    def __init__(self) -> None:
        pass


    def get_active_symbols(self, historical_data):
        max_date = str(historical_data['Date'].max())

        active = (historical_data[historical_data['Date'] == max_date])
        active = active['Symbol']

        list_unique_active = list(set(active))

        return list_unique_active
    


    def build_data_prep_models_file(self, cls_FileHandling, parameters):
        
        dados_indicators = cls_FileHandling.read_file(parameters.files_folder, parameters.file_w_indicators)

        active_symbols = self.get_active_symbols(dados_indicators)

        # Filter to clean data
        filtered_data = dados_indicators[dados_indicators['Symbol'].isin(active_symbols)]

        # Access dict with models configs
        _dict_config_train = parameters.cls_FileHandling.get_constants_dict(parameters, parameters.cls_Constants._get_configs_train(), default_config = 'v2.1.20')

        dados_prep = filtered_data[(filtered_data['Close'] != 0) & (filtered_data['Volume'] > _dict_config_train['min_volume_prep_models'])]

        if _dict_config_train['clean_targets_prep_models'] == True:
            dados_prep = dados_prep[(dados_prep['target_7d'] < 3) & (dados_prep['target_7d'] > - 0.9) & (dados_prep['target_15d'] < 3) & (dados_prep['target_15d'] > - 0.9) & (dados_prep['target_30d'] < 3) & (dados_prep['target_30d'] > - 0.9)]

        path_prep_models = Path(parameters.files_folder) / parameters.file_prep_models
        
        cls_FileHandling.save_parquet_file(dados_prep, path_prep_models)

        print(f"Parquet file with indicators prep models saved to {path_prep_models} with {len(dados_prep)} rows")

        cls_FileHandling.wait_for_file(path_prep_models)

        return dados_prep
