import json
from datetime import datetime
from typing import TypedDict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from agent.tools import execute_sql_query

class AgentState(TypedDict):
    question: str
    sql_query: Optional[str]
    db_results: Optional[str]
    answer: Optional[str]
    error: Optional[str]
    retry_count: int

def generate_sql_query(state: AgentState) -> AgentState:
    try:
        llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)
        
        sql_generation_prompt = PromptTemplate.from_template("""
        You are an expert at converting natural language questions into SQL queries for a PostgreSQL database.
        Your goal is to write a single, valid, and executable SQL query based on the user's question.

        Here is the database schema:

        - products (product_id INT, product_name TEXT, description TEXT)
        - sales (id INT, order_id INT, product_id INT, quantity INT, unit_price DECIMAL(10, 2), order_date TIMESTAMP)
        
        Relationships:
        - `sales.product_id` references `products.product_id`

        **Important Rules:**
        1. Only use the tables and columns provided above.
        2. Do not use functions that are not standard SQL, unless necessary for date operations (e.g., DATE_TRUNC, EXTRACT).
        3. If the query requires a join, use `JOIN`.
        4. If a date-related question is asked (e.g., 'last month', 'last 6 months'), use the `order_date` column in the `sales` table and the `NOW()` function or specific dates to filter the data.
        5. For multi-word product names, use ILIKE.
        
        Question: {question}

        Example query:
        SELECT COUNT(id) AS total_sales FROM sales;
        """)
        
        sql_chain = sql_generation_prompt | llm | StrOutputParser()
        sql_query = sql_chain.invoke({"question": state["question"]})

        return {"sql_query": sql_query.strip(), "retry_count": state["retry_count"]}

    except Exception as e:
        return {"error": f"SQL generation failed: {e}", "retry_count": state["retry_count"]}

def execute_db_query(state: AgentState) -> AgentState:
    try:
        # The tool automatically executes the query and handles the database connection
        result = execute_sql_query.invoke({"query": state["sql_query"]})

        if "error" in result:
            return {"error": result["error"], "retry_count": state["retry_count"] + 1}
        
        return {"db_results": json.dumps(result), "retry_count": state["retry_count"]}

    except Exception as e:
        return {"error": f"Database query failed: {e}", "retry_count": state["retry_count"] + 1}

def format_answer(state: AgentState) -> AgentState:
    try:
        llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0.5)

        answer_prompt = PromptTemplate.from_template("""
        You are a business analyst named Michael. Your task is to provide a clear, concise, and helpful summary of the data results to the user.
        
        Here are the user's question and the data you retrieved:
        - **Question:** {question}
        - **Data:** {db_results}
        
        Based on this, summarize the findings and provide a human-readable answer. Do not include any technical details like SQL code or column names. Focus on the business insight. If the data is empty, mention that no results were found.
        """)

        answer_chain = answer_prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({"question": state["question"], "db_results": state["db_results"]})
        
        return {"answer": answer}

    except Exception as e:
        return {"error": f"Answer formatting failed: {e}", "retry_count": state["retry_count"]}

def route_to_next_step(state: AgentState) -> str:
    if state.get("error") and state["retry_count"] < 3:
        print(f"🔄 Retrying... Attempt {state['retry_count']}/3")
        return "generate_sql_query"
    elif state.get("error") and state["retry_count"] >= 3:
        return "error_state"
    elif state.get("db_results"):
        return "format_answer"
    else:
        return "generate_sql_query"

def build_agent_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("generate_sql_query", generate_sql_query)
    graph_builder.add_node("execute_db_query", execute_db_query)
    graph_builder.add_node("format_answer", format_answer)

    graph_builder.add_edge(START, "generate_sql_query")
    graph_builder.add_edge("generate_sql_query", "execute_db_query")
    graph_builder.add_edge("execute_db_query", "format_answer")
    graph_builder.add_edge("format_answer", END)
    
    graph_builder.add_node("error_state", lambda s: {"error": "Failed after multiple retries."})
    graph_builder.add_edge("error_state", END)

    return graph_builder.compile()
