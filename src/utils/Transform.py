# from pathlib import Path
# import pandas as pd

# import warnings

# # from src.utils.Features import add_indicators



# warnings.filterwarnings("ignore")



# class DataTransform():
    
#     def __init__(self) -> None:
#         pass

    
#     def clean_date(self, dados_date):

#         dados_date['Date'] = pd.to_datetime(dados_date['Date'])
#         dados_date['Date'] = dados_date['Date'].dt.strftime('%Y-%m-%d')

#         return dados_date
    

#     def build_crypto_indicators(self, cls_FileHandling, parameters):
#         # Specify the input Parquet file path
#         input_path = Path(parameters.files_folder) / parameters.file_ingestion

#         # Read Parquet file into a Pandas DataFrame
#         # df = cls_FileHandling.read_parquet_to_dataframe(str(input_path))
#         df = cls_FileHandling.read_file(str(input_path))
        

        

#         print('1')

#         # Remove companies with discrepant numbers
#         company_code = 'MYRIA-USD'
#         df = df[df['Symbol'] != company_code]

#         print('2')

#         # Add indicators to the DataFrame
#         crypto_indicators_dataframe = parameters.cls_Features.add_indicators(df)
#         # crypto_indicators_dataframe = pd.read_csv(r'D:/Github/Forked/crypto_technical_analysis/data/historical_2file_w_indicators_test0240927.csv')

#         print('3')
#         print(len(crypto_indicators_dataframe.columns))
#         crypto_indicators_dataframe = crypto_indicators_dataframe.drop(['0', 'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26'], axis=1, errors='ignore')
#         print(len(crypto_indicators_dataframe.columns))

#         crypto_indicators_dataframe = self.clean_date(crypto_indicators_dataframe)
#         print('4')


#         if crypto_indicators_dataframe is not None:
#             # Specify the folder and file name for saving the Parquet file
#             output_path = Path(parameters.files_folder, parameters.file_w_indicators)

#             # Create the output folder if it doesn't exist
#             output_path.parent.mkdir(parents=True, exist_ok=True)
#             print('5')


#             # Save the DataFrame as a Parquet file
#             # cls_FileHandling.save_parquet_file(crypto_indicators_dataframe, output_path)
#             # crypto_indicators_dataframe.to_csv(output_path, index=False)
#             # print('6')

#             crypto_indicators_dataframe.to_parquet(output_path, compression = 'snappy')
#             print('7')



#             print(f"Parquet file saved to {output_path} with {len(crypto_indicators_dataframe)} rows")
            
#             cls_FileHandling.wait_for_file(output_path)

#         else:
#             print("No data fetched.")
