
from sqlalchemy import create_engine, inspect, text

# My own libraries or custom modules
import utils.db_util as db
import utils.indicators_util as iu

# Specify the name of the table in the database
engine = db.db_connection()

def main():
    # Use a context manager to handle the connection and automatically close it when done
    with engine.connect() as conn:
        if engine is None:
            return  # Exit the function if the connection is not established
        else:
            # Check if the specified table exists in the database
            table_exists = inspect(engine).has_table(table_name = 'crypto_indicators')    

            df = iu.add_indicators(db.get_db_data(conn, table_name = 'crypto_historical_price'))

            # If the table does not exist, create it
            if table_exists is False:
                db.create_raw_table(conn, table_name = 'crypto_indicators', data_type = 'indicators')
                print("Table has been created.")
                # Load the retrieved data into the database
                db.load_data_into_database(df, engine, table_name= 'crypto_indicators')   
            else:
                # If the table already exists, print a message
                print('Table exists')
                db.load_data_into_database(df, engine, table_name= 'crypto_indicators')   


# result
if __name__ == "__main__":
    main()
