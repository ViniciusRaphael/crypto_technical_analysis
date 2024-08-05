from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import duckdb
import time


class FileHandling():
      
    def __init__(self) -> None:
        pass


    def save_parquet_file(self, dataframe, file_path):
        """
        Save a Pandas DataFrame as a Parquet file.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame to be saved as a Parquet file.
        - file_path (str): Path where the Parquet file will be saved.
        """
        table = pa.Table.from_pandas(dataframe)
        pq.write_table(table=table, where=file_path, compression='snappy')

        
    def read_parquet_to_dataframe(self, parquet_path):
        """
        Read data from a Parquet file into a Pandas DataFrame.

        Parameters:
        - parquet_path (str): Path to the Parquet file.

        Returns:
        - pd.DataFrame: DataFrame containing the data from the Parquet file.
        """
        table = duckdb.read_parquet(parquet_path)
        return table.df()
    

    def read_file(self, folder_path, file_name):

        complete_path = Path(folder_path) / file_name

        if os.path.exists(complete_path) and os.stat(complete_path).st_size > 0:

            if file_name.split('.')[-1] == 'csv':
                file_content = pd.read_csv(complete_path)
                

            elif file_name.split('.')[-1] == 'parquet':
                # A função anterior read_parquet_to_dataframe levava um __index__ como coluna (pelo menos nessa nova estrutura)
                file_content = pd.read_parquet(complete_path)

            return file_content
        

    def wait_for_file(self, file_path):

        while not os.path.exists(file_path):
            time.sleep(0.2) 


    def get_selected_symbols(self, dados, parameters):

        return dados[dados['Symbol'].isin(parameters.filter_symbols)] if parameters.execute_filtered else dados
