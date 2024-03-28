# My own libraries or custom modules
import utils.db_util as db
import utils.indicators_util as iu
import matplotlib.pyplot as plt
import pandas as pd

engine = db.db_connection() 
with engine.connect() as conn:
    df = db.get_db_data(conn, table_name = 'crypto_indicators')
# df
    # df.set_index('date', inplace=True)
    # df['close'] = df['close'].astype('float')
df.info()
df.infer_objects().dtypes


# df.loc[df['symbol'] == 'BTC-USD']['close']
# df = pd.to_datetime(df.index)
# df_close = df[['close']]
# df_close
# print(df_close.index)  # Check the index values to see the available dates
# print(df_close.index.dtype)  # Check the data type of the index
# train = df_close[:'2023-12']
# test = df_close['2024-01':]
# pd.concat([train.add_suffix('_train')['Close_train'], test.add_suffix('_test')['Close_test']], axis=1, sort=False).plot()