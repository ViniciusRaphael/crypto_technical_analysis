from configparser import ConfigParser
from sqlalchemy import create_engine, inspect, text

import utils.db_util as db

# Read configuration from 'config.ini' file
config = ConfigParser()
config.read('../config.ini')

def main():
    # Specify the name of the table in the database
    table_name = 'crypto_historical_price'

    # Connect to the PostgreSQL database using the specified connection parameters
    engine = db.db_connection()

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
                db.create_raw_table(conn, data_type = 'raw', table_name=table_name)
                print("Table has been created.")
            else:
                # If the table already exists, print a message
                print('Table exists')

                # Retrieve data from the database using a specific query
                df = db.get_api_data(conn, table_name)

                # Load the retrieved data into the database
                db.load_data_into_database(df, engine, table_name)


if __name__ == "__main__":
    main()
