import mysql.connector as mysql

class DB():
    def __init__(self, database):
        self.db = mysql.connect(
            host = "localhost",
            user = "root",
            passwd = "12345",
            database = database
        )
        #self.cursor = self.db.cursor()
    def query(self, q):
        cursor = self.db.cursor()
        cursor.execute(q)
        return cursor.fetchall()