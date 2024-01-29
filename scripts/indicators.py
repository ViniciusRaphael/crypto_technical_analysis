# Python standard libraries
from configparser import ConfigParser

# Third-party libraries
import pandas as pd
import pandas_ta as ta
from sqlalchemy import text
import plotly.graph_objects as go
import time


# My own libraries or custom modules
import db_util as du

# Read configuration from 'config.ini' file
config = ConfigParser()
config.read('../config.ini')

def get_db_data(conn, table):
    """
    Retrieve data from a specific table in the database using the provided connection.

    Parameters:
    - conn: Database connection object.
    - table (str): The name of the table from which to retrieve data.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the specified table.
    """

    # Construct the SQL query to select all columns from the specified table
    sql_query = f"""
                    SELECT *
                    FROM {table}
                    """

    # Execute the SQL query using the provided connection
    output = conn.execute(text(sql_query))

    # Convert the query result to a DataFrame
    result = pd.DataFrame(output.fetchall())

    return result
def add_indicators(dataframe):
    """
    Add indicators to the DataFrame, including the 21-day Simple Moving Average (SMA) and the maximum of the last 55 days.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing financial data.

    Returns:
    - pd.DataFrame: The updated DataFrame with added indicators.
    """

    # Sorts the DataFrame by 'symbol' and 'date' to ensure proper application of operations
    dataframe = dataframe.sort_values(by=['symbol', 'date'])

    # Calculates and adds the 21-day Simple Moving Average (SMA) to the DataFrame
    dataframe['ma_21'] = dataframe.groupby('symbol')['close'].transform(lambda x: ta.sma(x, 21))

    # Calculates and adds the maximum of the last 55 days (excluding the current value) to the DataFrame
    dataframe['max_55'] = dataframe.groupby('symbol')['close'].transform(lambda x: x.shift(1).rolling(window=55).max())

    return dataframe


# Configuration parameters for the PostgreSQL database
db_connection = {
    'host': config.get('Database', 'host'),
    'port': config.getint('Database', 'port'),
    'database': config.get('Database', 'database'),
    'user': config.get('Database', 'user'),
    'password': config.get('Database', 'password')
}   

# Specify the name of the table in the database
table_name = 'crypto_historical_price'

engine = du.connect_to_database(db_connection)

# Use a context manager to handle the connection and automatically close it when done
with engine.connect() as conn:
    df = get_db_data(conn, table_name)

    # Ordenar o DataFrame por 'Nome do Grupo' e 'Data'
    start = time.time()
    df1 = add_indicators(df)
    end = time.time()
    print(f'Code finished in: {end - start} sec')

