from langchain.tools import Tool
import pandas as pd, matplotlib.pyplot as plt, numpy as np, io, contextlib, sys


def load_csv(filename: str) -> str:
    try:
        df = pd.read_csv(filename)
        globals()['df'] = df
        
        return f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns.\nColumns: {list(df.columns)}"
    except Exception as e:
        return f"Error loading file: {e}"


def get_dataframe_info(dummy_input: str = "") -> str:
    """Get information about the currently loaded DataFrame"""
    current_df = globals().get('df')
    if current_df is None:
        return "No DataFrame currently loaded. Please load a CSV file first."
    
    try:
        info_parts = []
        info_parts.append(f"DataFrame Shape: {current_df.shape}")
        info_parts.append(f"Columns: {list(current_df.columns)}")
        info_parts.append("\nColumn Data Types:")
        info_parts.append(str(current_df.dtypes))
        info_parts.append(f"\nFirst few rows:")
        info_parts.append(str(current_df.head(3)))
        
        # Check for null values
        null_counts = current_df.isnull().sum()
        if null_counts.sum() > 0:
            info_parts.append(f"\nNull values per column:")
            info_parts.append(str(null_counts[null_counts > 0]))
        else:
            info_parts.append(f"\nNo null values found.")
            
        return "\n".join(info_parts)
    except Exception as e:
        return f"Error getting DataFrame info: {e}"


def execute_code(code: str) -> str:
    # Ensure we have access to the global df
    current_df = globals().get('df')
    if current_df is None:
        return "Error: No DataFrame loaded. Please load a CSV file first."
    
    exec_locals = {
        "pd": pd, 
        "plt": plt, 
        "np": np, 
        "df": current_df  # Use the actual loaded DataFrame
    }

    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
            # Configure matplotlib for better output
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (10, 6)
            
            exec(code, {}, exec_locals)
            
            # Show any plots that were created
            if plt.get_fignums():
                plt.tight_layout()
                plt.show()
                
    except Exception as e:
        error_msg = error_buffer.getvalue()
        if error_msg:
            return f"Error executing code: {e}\nDetails: {error_msg}"
        else:
            return f"Error executing code: {e}"
    finally:
        plt.close('all')  # Clean up plots
    
    result = output_buffer.getvalue()
    error_output = error_buffer.getvalue()
    
    if error_output:
        result += f"\nWarnings/Errors: {error_output}"
    
    return result if result.strip() else "Code executed successfully (no output produced)"


load_csv_tool = Tool(
    name="load_csv",
    func=load_csv,
    description="Loads a CSV file into a DataFrame named 'df'.",
)

get_dataframe_info_tool = Tool(
    name="get_dataframe_info", 
    func=get_dataframe_info,
    description="Get detailed information about the currently loaded DataFrame including shape, columns, data types, and sample data.",
)

execute_code_tool = Tool(
    name="execute_code",
    func=execute_code,
    description="Executes provided Python code using pandas/matplotlib. Has access to 'df' DataFrame and common libraries.",
)

tools = [load_csv_tool, get_dataframe_info_tool, execute_code_tool]