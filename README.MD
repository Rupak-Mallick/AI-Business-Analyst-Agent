# Michael - The AI Business Analyst Agent

Michael is an AI-powered business analyst agent designed to answer business-related queries in natural language by interacting with a database. It uses a Large Language Model (LLM) to translate natural language questions into database queries, execute them, and return the results in a human-friendly format.

### 🎯 Features
- **Natural Language to SQL:** Translates user questions into executable PostgreSQL or MongoDB queries.
- **Data-driven Insights:** Retrieves and presents insights from structured business data.
- **Modular Design:** The project is structured into logical components for maintainability.

### 🛠️ Prerequisites
- Python 3.8+
- Docker & Docker Compose
- `pip` for installing Python packages

### ⚙️ Setup and Installation

1.  **Clone the Repository (or create the file structure):**
    First, set up the project directory as shown in the modular structure above.

2.  **Set up the Database with Docker Compose:**
    Navigate to the project root directory and run the Docker Compose file to start the PostgreSQL and MongoDB containers.

    ```bash
    docker compose -f docker-compose-infra.yml up -d
    ```

3.  **Install Python Dependencies:**
    Create a virtual environment and install the required packages.

    ```bash
    pip install -r requirements.txt
    ```

    *Note: You will need to create a `requirements.txt` file based on the packages used in your Jupyter notebooks, such as `psycopg2-binary`, `pymongo`, `langchain`, `langgraph`, and `langchain-groq`.*

4.  **Insert Sample Data:**
    Once the database containers are running, execute the data insertion script for your chosen database (e.g., PostgreSQL).

    ```bash
    python database/data/insert_data_pgsql.py
    ```
    or for MongoDB
    ```bash
    python database/data/insert_data_mongo.py
    ```

### 🏃 How to Run the Agent

1.  **Configure API Key:**
    Before running the main script, ensure you have your API key for the chosen LLM (e.g., GROQ) set as an environment variable.

    ```bash
    export GROQ_API_KEY="your_api_key_here"
    ```

2.  **Start the Agent:**
    Run the `main.py` script to start the agent and interact with it.

    ```bash
    python main.py
    ```

### 📂 Code Breakdown

- **`database/`:** Contains all database-specific logic. The `postgres_db.py` handles the connection and query execution for PostgreSQL, keeping database interactions separate from the agent's core logic.
- **`agent/`:** Contains the core LangGraph agent logic. `agent_core.py` defines the state machine, and `tools.py` defines the functions the agent can call, such as `execute_sql_query`.
- **`config/settings.py`:** A central place for all your configurations and API keys, making them easy to manage.
- **`main.py`:** This is the entry point of the application. It initializes the agent and the database connection and contains the main loop for user interaction.