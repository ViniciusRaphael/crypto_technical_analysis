# My own libraries or custom modules
import utils.db_util as db
import utils.indicators_util as iu
from datetime import datetime, timedelta

engine = db.db_connection() 
with engine.connect() as conn:
    df = db.get_db_data(conn, table_name = 'crypto_indicators')
    yesterday = str(datetime.today().date() - timedelta(days = 1))
    result = iu.filter_daily_indicators(dataframe=df, date = yesterday)

result