from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_functions_agent
from analyticTools import load_csv_tool, execute_code_tool, get_dataframe_info_tool
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful data analysis assistant. A DataFrame named 'df' is already loaded and available for analysis.

You can handle various types of questions including:
- Relationships between variables (correlation, comparison)  
- Time-based analysis (when do events peak, trends over time)
- Aggregation queries (highest, lowest, most frequent values)
- Statistical summaries and distributions

IMPORTANT: The DataFrame 'df' is already loaded and ready to use. Do NOT ask for CSV files.

Follow these steps for ANY analysis:
1. Restate what the user wants in clear terms.
2. Use get_dataframe_info tool first to understand the current DataFrame structure if needed.
3. Identify the relevant DataFrame columns needed for the analysis.
4. Explain how to handle null values if present.
5. Describe the analysis approach:
   - For "when" questions: Group by time/categorical variables and aggregate
   - For "most/highest/peak" questions: Use groupby + agg functions, then find max/min
   - For relationships: Use correlation, comparison, or visualization
   - For trends: Sort by time/sequence and analyze patterns
6. Specify which chart or output to produce and why it's appropriate.
7. Generate Python code (with pandas/matplotlib) that performs the analysis.
8. If you can't find a column in the dataframe use the column that is related to it the most 
9. You must always write the visualization code  

Important code patterns:
- For time-based questions: Use df.groupby() with time columns, then .sum(), .count(), .mean(), etc.
- For finding peaks/maximums: Use .idxmax(), .max(), or .sort_values(ascending=False)
- For "most customers": Count occurrences with .value_counts() or .groupby().size()
- Each row typically represents one customer/transaction, so count rows for customer count

Always wrap the Python code in triple backticks after your explanation.
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(
    llm=llm,
    tools=[load_csv_tool, execute_code_tool, get_dataframe_info_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[load_csv_tool, execute_code_tool, get_dataframe_info_tool],
    verbose=True,
    handle_parsing_errors=True
)

# Store chat history for context
chat_history = []

# Example main loop:
print("Start data analysis. (type 'exit' to quit)\n")
newChat = True
isCorrected = False
while True:
    if not isCorrected:
        if newChat:
            user_input = input("Please enter the name of csv file: ")
        else: 
            user_input = input("Please enter your query: ")
        
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # First, if input mentions a file, load it:
    if "csv" in user_input.lower() and newChat:
        filename = user_input.strip()
        print("Loading file:", filename)
        result = load_csv_tool.run(filename)
        print(result)
        
        # Show basic info about the dataset
        info_result = get_dataframe_info_tool.run("")
        print("\nDataset Overview:")
        print(info_result)
        
        # Add to chat history
        chat_history.append(("human", f"Load CSV file: {filename}"))
        chat_history.append(("ai", f"Loaded successfully. {result}\n{info_result}"))
        
        newChat = False
        continue
    
    # Get plan & code from agent with chat history
    raw = agent_executor.invoke({
        "query": user_input,
        "chat_history": chat_history
    })
    
    ai_output = raw["output"]
    
    # Add to chat history
    chat_history.append(("human", user_input))
    chat_history.append(("ai", ai_output))
    
    if "```" in ai_output:
        parts = ai_output.split("```")
        explanation = parts[0].strip()
        if len(parts) >= 2:
            code_block = parts[1]
            code = code_block.replace("python", "").strip("` \n")
        else:
            code = ""
    else:
        explanation = ai_output
        code = ""
    
    print("\nAI Plan and Explanation:\n", explanation)
    
    if code:
        confirm = input("\nExecute this code? (yes/no): ").strip().lower()
        if confirm in ["yes", "y"]:
            isCorrected = False
            print("\nExecuting the generated code...")
            # Clean the code
            if "pd.read_csv" in code:
                code = "\n".join([line for line in code.splitlines() if "read_csv" not in line])
            
            result = execute_code_tool.run(code)
            print("\nExecution Result:\n", result)
        else:
            correction = input("Please provide corrected instructions: ")
            # Replace the last entry with correction
            isCorrected = True
            chat_history[-2] = ("human", correction)
            continue
    else:
        input("\nPress Enter to continue...")