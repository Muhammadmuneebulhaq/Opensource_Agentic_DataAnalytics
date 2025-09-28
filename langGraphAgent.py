import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib
from dotenv import load_dotenv
import os

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Global DataFrame
df = None

# ---------------- TOOLS ---------------- #

@tool
def load_csv_tool(filename: str) -> str:
    """Load a CSV file into the global DataFrame."""
    global df
    try:
        df = pd.read_csv(filename)
        return f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns."
    except Exception as e:
        return f"Error loading file: {e}"

@tool
def get_dataframe_info_tool(_: str = "") -> str:
    """Return info about the DataFrame: columns, datatypes, null counts."""
    global df
    if df is None:
        return "No DataFrame loaded."
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

@tool
def execute_code_tool(code: str) -> str:
    """Execute Python code that uses pandas/matplotlib with the global df."""
    global df
    local_vars = {"df": df, "pd": pd, "plt": plt}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {}, local_vars)
        return "Execution successful.\n" + output.getvalue()
    except Exception as e:
        return f"Execution error: {e}"

# ---------------- LLM + PROMPT ---------------- #

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a data analysis assistant. A pandas DataFrame named 'df' is available.  
You can use the following tools:
- load_csv_tool
- get_dataframe_info_tool
- execute_code_tool

Rules:
1. Do NOT ask for CSV unless user explicitly provides a filename.
2. Always check df structure with get_dataframe_info_tool if needed.
3. For analysis: explain the steps, then provide pandas/matplotlib code.
4. Always wrap code in triple backticks.
5. Always attempt to use execute_code_tool for running code.
6. If a column isn't found, suggest the closest related column.
    """),
    ("human", "{input}")
])

# Create ReAct Agent
agent_executor = create_react_agent(llm, tools=[load_csv_tool, get_dataframe_info_tool, execute_code_tool])

# ---------------- MAIN LOOP ---------------- #

def main():
    print("Start data analysis. (type 'exit' to quit)")

    filename = input("Please enter the name of csv file : ")
    df = pd.read_csv(filename)
    print(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns.\n")

    while True:
        query = input("Please enter your query: ")
        if query.lower() == "exit":
            break
        if not query.strip():  # prevent empty query
            print("⚠️ Please enter a valid query.")
            continue

        try:
            response = agent_executor.invoke({
                "messages": [
                    {"role": "user", "content": f"Use df to answer: {query}"}
                ]
            })

            print(response)  
        except Exception as e:
            print("❌ Error:", e)

if __name__ == "__main__":
    main()
