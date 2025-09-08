import psycopg2
from psycopg2.extras import RealDictCursor
from config.settings import POSTGRES_URI

class PostgreSQLDB:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(POSTGRES_URI)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("✅ Successfully connected to PostgreSQL database.")
        except psycopg2.OperationalError as e:
            print(f"❌ Could not connect to PostgreSQL: {e}")
            self.conn = None
            self.cursor = None

    def execute_query(self, query):
        if not self.conn or self.conn.closed:
            self._connect()
        if not self.cursor:
            return {"error": "Database connection not available."}

        try:
            self.cursor.execute(query)
            if query.strip().lower().startswith("select"):
                results = self.cursor.fetchall()
                # LangGraph doesn't handle RealDictCursor objects well, so convert to dict
                return [dict(row) for row in results]
            else:
                self.conn.commit()
                return {"message": "Query executed successfully."}
        except psycopg2.Error as e:
            self.conn.rollback()
            return {"error": f"SQL query failed: {e}"}

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
