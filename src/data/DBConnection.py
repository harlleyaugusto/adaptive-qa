import mysql.connector as mysql
import pandas as pd
import src.util.config as c

class DB():

    def __init__(self, database):
        config = c.Config()

        self.db = mysql.connect(
            host = config.host,
            user = config.user,
            passwd = config.password,
            database = database
        )

    def query(self, q):
        cursor = self.db.cursor()
        cursor.execute(q)
        return cursor.fetchall()

if __name__ == '__main__':
    # Just testing
    db = DB('cook')
    q_a = pd.read_sql('SELECT DISTINCT id, parentId FROM Posts', db.db)