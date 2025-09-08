import json
import os
from datetime import datetime
from agent.agent_core import build_agent_graph
from database.postgres_db import PostgreSQLDB
from database.mongo_db import MongoDB
from dotenv import load_dotenv

load_dotenv()

def process_query(question: str) -> str:
    """
    Processes a single natural language question and returns the answer.
    """
    agent = build_agent_graph()
    inputs = {"question": question, "retry_count": 0}
    
    final_state = None
    for output in agent.stream(inputs):
        for key, value in output.items():
            print(f"Current state: {key}")
            # print(f"Output from {key}: {value}\n")
        final_state = output

    if 'format_answer' in final_state and 'answer' in final_state['format_answer']:
        return final_state['format_answer']['answer']
    
    return final_state.get('error', 'No answer generated')

def main():
    """
    Main function to run the business analyst agent.
    """
    db_choice = os.getenv("DB_CHOICE", "postgres").lower()

    if db_choice == "postgres":
        db = PostgreSQLDB()
    elif db_choice == "mongo":
        db = MongoDB()
    else:
        print("Invalid DB_CHOICE. Using PostgreSQL as default.")
        db = PostgreSQLDB()

    if not db or not (hasattr(db, 'conn') and db.conn) and not (hasattr(db, 'client') and db.client):
        print("Exiting due to failed database connection.")
        return

    queries = [
        "What is the total sales revenue in 2025?",
        "Which product sold the most units?",
        "Summarize sales trends for March 2025."
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = process_query(query)
        print(f"\nResponse: {response}\n")

    db.close()

if __name__ == "__main__":
    main()
