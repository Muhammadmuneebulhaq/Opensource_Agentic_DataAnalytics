from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

# Load environment variables (for API keys)
load_dotenv()

# Import tools from analyticTools (assumes analyticTools.py is available)
from analyticTools import load_csv_tool, get_dataframe_info_tool, execute_code_tool

# Define the system prompt with detailed instructions
system_prompt = """You are a helpful data analysis assistant. You have access to the following tools:
- load_csv: Loads a CSV file into a DataFrame named 'df'.
- get_dataframe_info: Provides detailed information about the loaded DataFrame (shape, columns, data types, sample data, null values).
- execute_code: Executes provided Python code using pandas and matplotlib; it has access to the DataFrame 'df', pandas (pd), matplotlib.pyplot (plt), and numpy (np).

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

# Initialize the chat model and bind tools for the agent
# Ensure to set your API key in the OPENAI_API_KEY environment variable
agent_model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0).bind_tools(
    [load_csv_tool, get_dataframe_info_tool]
)

# Define the node function to call the agent model
def call_agent_node(state: MessagesState) -> dict[str, list[AIMessage]]:
    # Invoke the chat model with the conversation history
    response = agent_model.invoke(state["messages"])
    return {"messages": [response]}

# Define routing function to decide when to call the tool node
def route_tools(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "data_tools"
    return END

# Create a ToolNode with the analysis tools (using the tool functions)
tool_node = ToolNode([load_csv_tool, get_dataframe_info_tool, execute_code_tool])

# Build the state graph with two nodes: agent and tools
workflow = StateGraph(MessagesState)
workflow.add_node("analysis_agent", call_agent_node)
workflow.add_node("data_tools", tool_node)
workflow.add_edge(START, "analysis_agent")
workflow.add_conditional_edges("analysis_agent", route_tools, ["data_tools", END])
workflow.add_edge("data_tools", "analysis_agent")
analysis_graph = workflow.compile()

# Store the conversation messages (with system prompt as first message)
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
