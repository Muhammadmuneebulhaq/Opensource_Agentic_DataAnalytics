from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

# Load environment variables (for API keys)
load_dotenv()

# Import tools from analyticTools (assumes analyticTools.py is available)
from analyticTools import (
    load_csv_tool, 
    get_dataframe_info_tool, 
    execute_code_tool,
    ingest_source_tool,
    embed_documents_tool,
    search_documents_tool
)

# Define the system prompt with detailed instructions
system_prompt = """You are a helpful data analysis assistant. You have access to the following tools:
- load_csv: Loads a CSV file into a DataFrame named 'df'.
- get_dataframe_info: Provides detailed information about the loaded DataFrame (shape, columns, data types, sample data, null values).
- execute_code: Executes provided Python code using pandas and matplotlib; it has access to the DataFrame 'df', pandas (pd), matplotlib.pyplot (plt), and numpy (np).
- ingest_source: Ingest data from CSV, Excel, JSON, PDF, or database connection strings.
- embed_documents: Embed all loaded documents (PDFs, text) into ChromaDB for semantic search.
- search_documents: Search embedded documents and return top 3 relevant chunks with similarity scores.

If the user provides a CSV filename (e.g., 'data.csv') before any data is loaded, call load_csv with the filename. Then, immediately call get_dataframe_info to display the dataset overview to the user.

For any data analysis query, follow these steps:
1. Restate the user’s request.
2. Use get_dataframe_info if needed to understand the DataFrame structure.
3. Identify relevant columns for the analysis.
4. Explain how to handle null values.
5. Describe the analysis approach:
   - For time-based questions: group by time and aggregate (sum, count, mean, etc.).
   - For 'most/highest/peak' questions: use groupby + agg, then find max/min.
   - For relationships: use correlation, comparison, or visualization.
   - For trends: sort by time or sequence and analyze patterns.
6. Specify which chart or output to produce and why.
7. Generate Python code (pandas/matplotlib) to perform the analysis.
8. If the user mentions a column not in df, use a closely related column instead.
9. Always include visualization code (plt).

Important code patterns:
- For time questions: df.groupby(time_column).sum(), count(), or mean().
- For peaks: .idxmax(), .max(), or .sort_values().
- For counting: use df['col'].value_counts() or df.groupby(...).size().
- Each row typically represents one record, so count rows for totals.

Wrap the Python code in triple backticks after your explanation."""

# Initialize the chat model with tools bound
# Ensure to set your API key in the OPENAI_API_KEY environment variable
agent_model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0).bind_tools(
    [load_csv_tool, 
     get_dataframe_info_tool, 
     execute_code_tool,
     ingest_source_tool,
     embed_documents_tool,
     search_documents_tool]
)

# Map tool names to actual tool functions
tool_map = {
    "load_csv": load_csv_tool.func,
    "get_dataframe_info": get_dataframe_info_tool.func,
    "execute_code": execute_code_tool.func,
    "ingest_source": ingest_source_tool.func,
    "embed_documents": embed_documents_tool.func,
    "search_documents": search_documents_tool.func,
}

# Agent node - calls the model
def agent(state):
    """Agent node that calls Claude to process messages."""
    messages = state.get("messages", [])
    response = agent_model.invoke(messages)
    return {"messages": [response]}

# Tool node - executes tool calls
def tools(state):
    """Tool node that executes any tool calls from the agent."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}
    
    tool_calls = last_message.tool_calls
    tool_results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name") or tool_call.get("type")
        tool_input = tool_call.get("args", {})
        tool_id = tool_call.get("id")
        
        # Execute the tool
        if tool_name in tool_map:
            try:
                if isinstance(tool_input, dict):
                    result = tool_map[tool_name](**tool_input)
                else:
                    result = tool_map[tool_name](tool_input)
            except Exception as e:
                result = f"Error executing {tool_name}: {str(e)}"
        else:
            result = f"Unknown tool: {tool_name}"
        
        # Create tool message
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_id,
            name=tool_name
        )
        tool_results.append(tool_message)
    
    return {"messages": tool_results}

# Routing function
def route_after_agent(state):
    """Route based on whether the agent called tools."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Build the graph
class AgentState(TypedDict):
    messages: Annotated[list, add]

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

analysis_graph = workflow.compile()

# For backward compatibility
def invoke_agent(messages):
    """Invoke the agent with a list of messages."""
    if not messages or messages[0].__class__.__name__ != "SystemMessage":
        messages = [SystemMessage(content=system_prompt)] + messages
    result = analysis_graph.invoke({"messages": messages})
    return result

# Store the conversation messages (with system prompt as first message)
# (Only for CLI usage)

if __name__ == "__main__":
    conversation_messages = [SystemMessage(content=system_prompt)]

    # CLI loop for user interaction
    print("Start data analysis. (type 'exit' to quit)\n")
    csv_loaded = False

    while True:
        try:
            if not csv_loaded:
                user_input = input("Please enter the name of csv file: ").strip()
            else:
                user_input = input("Please enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in ["exit", "quit"]:
            break

        # Append user's message to conversation
        conversation_messages.append(HumanMessage(content=user_input))

        # If CSV is not loaded yet, process loading
        if not csv_loaded:
            state = {"messages": conversation_messages}
            state = analysis_graph.invoke(state)
            conversation_messages = state["messages"]
            last_msg = conversation_messages[-1]
            ai_output = last_msg.content
            print("\n" + ai_output)
            csv_loaded = True
            continue

        # Analysis query processing
        while True:
            state = {"messages": conversation_messages}
            state = analysis_graph.invoke(state)
            conversation_messages = state["messages"]
            last_msg = conversation_messages[-1]
            ai_output = last_msg.content

            # Parse explanation and code from output
            if "```" in ai_output:
                parts = ai_output.split("```")
                explanation = parts[0].strip()
                code = ""
                if len(parts) >= 2:
                    code_block = parts[1]
                    code = code_block.replace("python", "").strip("` \n")
            else:
                explanation = ai_output.strip()
                code = ""

            # Display the AI's plan and explanation
            print("\nAI Plan and Explanation:\n", explanation)
            if code:
                print("\nAI Generated Code:\n```python\n" + code + "\n```")

                confirm = input("\nExecute this code? (yes/no): ").strip().lower()
                if confirm in ["yes", "y"]:
                    print("\nExecuting the generated code...")
                    # Remove any CSV loading lines to avoid reloading
                    code_lines = code.splitlines()
                    code_cleaned = "\n".join([line for line in code_lines if "read_csv" not in line])
                    result = execute_code_tool.run(code_cleaned)
                    print("\nExecution Result:\n", result)
                    break  # finished this query
                else:
                    correction = input("\nPlease provide corrected instructions: ").strip()
                    conversation_messages.append(HumanMessage(content=correction))
                    # Loop again to process correction with the updated conversation
                    continue
            else:
                # No code to execute, end this query
                break

        # Continue to next user query

    # End of agent session
    print("Session ended.")
