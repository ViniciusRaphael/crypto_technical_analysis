# Python standard libraries
import time
from datetime import timedelta
from configparser import ConfigParser

# Third-party libraries
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, inspect, text

# My own libraries or custom modules
import crypto_symbols_util as cs

# Read configuration from 'config.ini' file
config = ConfigParser()
config.read('../config.ini')

# Get a list of cryptocurrency symbols from Kucoin
kucoin_list = cs.get_kucoin_symbols()
# Remove the last character from each symbol in the list
# USDT to USD - name convention on yfinance
crypto_list = [i[:-1] for i in kucoin_list]

def connect_to_database(connection_params):
    """
    Connect to the PostgreSQL database.
    
    Parameters:
    - connection_params (dict): Dictionary containing database connection parameters.

    Returns:
    - sqlalchemy.engine.Engine: SQLAlchemy engine for database connection.
    """
    try:
        engine = create_engine(f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")
        print('Connected to the database')
        return engine
    except Exception as e:
        print(f'Error: {e}')
        return None
    
def create_table(conn, table_name):
    """
    Create a table in the database.

    Parameters:
    - conn: SQLAlchemy database connection.
    - table_name (str): Name of the table to be created.
    """    
    sql_code = f"""CREATE TABLE {table_name}(
                    date DATE,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume BIGINT,
                    dividends NUMERIC,
                    symbol VARCHAR(50))
                """
    # Execute a command: create datacamp_courses table
    conn.execute(text(sql_code))

    # Make the changes to the database persistent
    conn.commit()
    conn.close()

def get_api_data(conn, table):
    sql_query = f"""
                    SELECT *
                    FROM (
                            SELECT 
                            *,
                            ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY date DESC) AS qualify
                        FROM {table}
                        ) AS p
                    WHERE p.qualify = 1
                    """
    output = conn.execute(text(sql_query))
    result = pd.DataFrame(output.fetchall())
    # If the table doesn't exist, create dataframes for historical cryptocurrency data
    if result.empty:
        return download_all_cryptos()
        
    else:
        return update_cryptos(dataframe = result)
       

def download_all_cryptos(crypto_list=crypto_list, timesleep=1):
    """
    Downloads historical data for a list of cryptocurrencies using yfinance.

    Parameters:
    - crypto_list (list): List of cryptocurrencies to download.
    - timesleep (int): Time in seconds to sleep between API calls.

    Returns:
    DataFrame containing historical data for all the cryptocurrencies.
    """
    # Initialize an empty list to store dataframes for each cryptocurrency
    dataframes = []
    
    # Counter to keep track of the iteration
    count = 0
    
    # Loop through the list of cryptocurrencies
    for crypto in crypto_list:
        
        # Print the current cryptocurrency being processed
        print(crypto)

        # Get historical data using yfinance for the given symbol since January 1, 2020
        crypto_data = yf.Ticker(f"{crypto}").history(start="2020-01-01")
        
        # Add a 'symbol' column to the dataframe to identify the cryptocurrency
        crypto_data['symbol'] = crypto
        
        # Increment the counter and print progress
        count += 1
        print(f'Item {count} of {len(crypto_list)}')
        
        # Add data to the dataframe if it's not empty
        if not crypto_data.empty:
            # Append the dataframe to the list
            dataframes.append(crypto_data.reset_index())
            
            # Pause execution for a specified duration to avoid overloading the server
            time.sleep(timesleep)

    # Concatenate all dataframes into a single dataframe
    df = pd.concat(dataframes)
    
    # Drop the 'Stock Splits' column as it's not needed
    df.drop(columns='Stock Splits', inplace=True)
    
    # Convert column names to lowercase for consistency
    df.columns = df.columns.str.lower()

    # Return the final dataframe
    return df

def update_cryptos(dataframe, crypto_list=crypto_list, timesleep=1):
    """
    Updates cryptocurrency data by fetching historical data for each cryptocurrency in the provided list.
    
    Parameters:
    - dataframe (DataFrame): Existing cryptocurrency data.
    - crypto_list (list): List of cryptocurrencies to update.
    - timesleep (int): Time in seconds to sleep between API calls.
    
    Returns:
    List of DataFrames containing updated cryptocurrency data.
    """
    # List to store dataframes for each cryptocurrency
    empty_dataframe = []
    
    # Counter to keep track of the iteration
    count = 0
    
    # Loop through the list of cryptocurrencies
    for count, crypto in enumerate(crypto_list, start=1):
        print(f'Processing {crypto} ({count} of {len(crypto_list)})')

        # Check if the symbol is already present in the existing data
        if crypto in dataframe['symbol'].values:
            print(f'{crypto} found in existing data.')

            # Get the maximum date for the given cryptocurrency in the existing data
            max_date = dataframe[dataframe['symbol'] == crypto]['date'].max()

            start_date = max_date + timedelta(days=1)

            # Convert max_date to datetime if needed
            start_date = pd.to_datetime(start_date) if start_date is not pd.NaT else pd.to_datetime('2020-01-01')
            
            # Get historical data from the last recorded date in the existing data
            crypto_data = yf.Ticker(crypto).history(start=start_date)
            crypto_data['symbol'] = crypto
            # Check if the retrieved data is not empty before appending to empty_dataframe
            if not crypto_data.empty:
                empty_dataframe.append(crypto_data)
                time.sleep(timesleep)
        else:
            print(f'{crypto} not found in existing data.')
            # Get historical data from the start date if the cryptocurrency is not in existing data
            crypto_data = yf.Ticker(crypto).history(start='2020-01-01')
            crypto_data['symbol'] = crypto
            # Check if the retrieved data is not empty before appending to empty_dataframe
            if not crypto_data.empty:
                empty_dataframe.append(crypto_data)
                time.sleep(timesleep)

    dataframe_concat = pd.concat(empty_dataframe)
    dataframes = pd.DataFrame(dataframe_concat).reset_index()
    # Drop the 'Stock Splits' column as it's not needed
    dataframes.drop(columns='Stock Splits', inplace=True)
    dataframes.columns = dataframes.columns.str.lower()
    # Return the list of dataframes
    return dataframes

def load_data_into_database(df, engine, table_name, if_exists='append'):
    """
    Load DataFrame data into the PostgreSQL database.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing data to be loaded.
    - engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.
    - table_name (str): Name of the table to which data will be loaded.
    - if_exists (str): Specifies behavior when the table already exists. Default is 'append'.

    Returns:
    - None
    """

    # Check if DataFrame is empty
    if df.empty:
        print('DataFrame is empty. No data to load.')
        return

    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        print(f'DataFrame data loaded into {table_name} table')
    except Exception as e:
        print(f'Error loading data into the database: {e}')

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

def main():
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

    # Uncomment and modify the following line if reading data from a CSV file
    # df = read_csv(csv_file_path)

    # Connect to the PostgreSQL database using the specified connection parameters
    engine = connect_to_database(db_connection)

    # Use a context manager to handle the connection and automatically close it when done
    with engine.connect() as conn:
        # Check if the engine is successfully created
        if engine is None:
            return  # Exit the function if the connection is not established
        else:
            # Check if the specified table exists in the database
            table_exists = inspect(engine).has_table(table_name)
            
            # If the table does not exist, create it
            if table_exists is False:
                create_table(conn, table_name)
                print("Table has been created.")
            else:
                # If the table already exists, print a message
                print('Table exists')
                
                # Retrieve data from the database using a specific query
                df = get_api_data(conn, table_name)
                
                # Load the retrieved data into the database
                load_data_into_database(df, engine, table_name)

    
if __name__ == "__main__":
    main()
