import duckdb
from pathlib import Path
import pandas as pd


def query_data(dataframe):
    # dataframe = dataframe
    result = duckdb.sql(f"""
                    SELECT
                        Symbol,
                        Close,
                        min_50,
                        percent_risk,
                        qt_days_tendency_positive,
                        qt_days_macd_delta_positive

                    FROM '{dataframe}'
                    WHERE 
                        1 = 1
                        AND Date = CURRENT_DATE()
                        AND vl_adx >= 25
                        AND vl_macd > vl_macd_signal
                        AND vl_macd_delta > 0.01
                        """)
    
    return result

def main():
    # Specify the input Parquet file path
    input_folder = 'files'
    input_file = 'crypto_data_with_indicators.parquet'
    input_path = Path(input_folder) / input_file

    # # Read Parquet file into a Pandas DataFrame
    # df = read_parquet_to_dataframe(str(input_path))
    # print(df)
    result = query_data(str(input_path))
    return result.df()

if __name__ == '__main__':
    result = main()