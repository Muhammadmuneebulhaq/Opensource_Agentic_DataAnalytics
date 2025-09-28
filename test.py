import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import io
import contextlib
from dotenv import load_dotenv
import os
load_dotenv()


# Initialize OpenAI client
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# Load CSV file
df = pd.read_csv("tips.csv")

def ask_ai_for_plan(user_request):
    """
    Step 1: Ask GPT to restate the request, 
    specify columns, null handling, and chart type.
    """
    prompt = f"""
    You are a helpful data analysis assistant. The user asked {user_request}A DataFrame named 'df' is already loaded and available for analysis.

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

Important code patterns:
- For time-based questions: Use df.groupby() with time columns, then .sum(), .count(), .mean(), etc.
- For finding peaks/maximums: Use .idxmax(), .max(), or .sort_values(ascending=False)
- For "most customers": Count occurrences with .value_counts() or .groupby().size()
- Each row typically represents one customer/transaction, so count rows for customer count

Always wrap the Python code in triple backticks after your explanation.
    """

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a helpful AI that chats and analyzes CSV data."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def execute_code(code):
    """
    Execute the generated Python code safely.
    """
    exec_locals = {"df": df, "plt": plt, "pd": pd}
    output_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):
        try:
            exec(code, {}, exec_locals)
        except Exception as e:
            print("Error executing code:", e)

    print("\nAnalysis Output:\n")
    print(output_buffer.getvalue())
    plt.show()


def chat_with_ai(user_request):
    while True:
        ai_reply = ask_ai_for_plan(user_request)

        # Split confirmation/explanation vs code
        if "```" in ai_reply:
            explanation, code = ai_reply.split("```", 1)
            code = code.replace("python", "").strip("` \n")
        else:
            explanation = ai_reply
            code = ""

        # Step 2: Show AI’s understanding
        print("\nAI Understanding:\n")
        print(explanation.strip())

        # Step 3: Ask user for confirmation
        confirm = input("\nDo you confirm this understanding? (yes/no): ").strip().lower()

        if confirm in ["yes", "y"]:
            if code:
                print("\nExecuting AI-selected analysis...\n")
                execute_code(code)
            break
        else:
            # Get corrections from user
            correction = input("\nPlease tell me the corrections: ")
            user_request = correction  # feed correction back into loop


# ---------------- MAIN LOOP ----------------
print("CSV loaded. Start chatting with the AI! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    chat_with_ai(user_input)
