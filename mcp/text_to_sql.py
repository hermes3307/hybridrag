import json
import re
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sqlalchemy import create_engine, inspect

# Database connection
db_url = "sqlite:///C://Users//A//Documents//cursor//hybridrag//ollama//testdb.sqlite"

# SQL generation template
template = """
You are a SQL generator. When given a schema and a user question, you MUST output only the SQL statement—nothing else. No explanation is needed.

Schema: {schema}
User question: {query}
Output (SQL only):
"""

# Streamlit UI setup
st.title("Text to SQL Generator")
st.write("Convert natural language to SQL queries")

# Schema extraction function
def extract_schema(db_url):
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        schema = {}

        for table in inspector.get_table_names():
            columns = inspector.get_columns(table)
            schema[table] = [col['name'] for col in columns]

        return json.dumps(schema)
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return "{}"

# Clean text function
def clean_text(text: str):
    if text is None:
        return ""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# Load schema
try:
    schema = extract_schema(db_url)
    st.write("Database schema loaded successfully")
except Exception as e:
    st.error(f"Error loading schema: {str(e)}")
    schema = "{}"

# Ollama connection status check
ollama_status = st.empty()
try:
    # Try to initialize the model with a timeout
    model = OllamaLLM(model="deepseek-r1:8b", base_url="http://localhost:11434")
    # Test connection with a simple generation
    model.invoke("test")
    ollama_status.success("Connected to Ollama service")
except Exception as e:
    ollama_status.error(f"Could not connect to Ollama: {str(e)}")
    st.warning("Make sure Ollama is running on your machine (default URL: http://localhost:11434)")
    model = None

# SQL generation function
def to_sql_query(query, schema):
    if model is None:
        return "Error: Ollama service not available"
    
    try:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        result = chain.invoke({"query": query, "schema": schema})
        return clean_text(result)
    except Exception as e:
        return f"Error generating SQL: {str(e)}"

# User input
query = st.text_area("Describe the data you want to retrieve from the database:")

# Generate SQL when user inputs a query
if query and st.button("Generate SQL"):
    with st.spinner("Generating SQL..."):
        sql = to_sql_query(query, schema)
        st.code(sql, language="sql")

# Optional: Add section to show database schema
if st.checkbox("Show Database Schema"):
    try:
        schema_dict = json.loads(schema)
        for table, columns in schema_dict.items():
            st.subheader(f"Table: {table}")
            st.write(", ".join(columns))
    except:
        st.error("Could not display schema")