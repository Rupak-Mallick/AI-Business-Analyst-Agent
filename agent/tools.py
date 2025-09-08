from langchain_core.tools import tool
from typing import Dict, Any, List
from database.postgres_db import PostgreSQLDB

@tool
def execute_sql_query(query: str) -> List[Dict[str, Any]] or Dict[str, str]:
    """
    Executes a given SQL query on the PostgreSQL database and returns the results.
    The query must be a valid SQL statement.
    """
    db = PostgreSQLDB()
    result = db.execute_query(query)
    db.close()
    return result
