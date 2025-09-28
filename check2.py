import pandas as pd

# List of CSV files to check (update with your actual file names)
files = [
    "WA_Fn-UseC_-HR-Employee-Attrition.csv",  # IBM HR Analytics Attrition Dataset
    "supply_chain_data.csv",                 # Supply Chain Analysis
    "smart_logistics_dataset.csv"            # Smart Logistics Supply Chain Dataset
]

# Loop through each file
for file in files:
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        print(f"\nChecking {file}:")
        
        # 1. Check Data Type Conversion
        print("\nData Type Conversion Check:")
        for column in df.columns:
            if df[column].dtype == 'object':  # Only check string columns
                # Try converting to numeric to detect hidden numbers
                try:
                    pd.to_numeric(df[column], errors='coerce')
                    null_count = df[column].isna().sum()
                    if null_count > 0:
                        print(f"  {column} may need conversion (e.g., text to numeric), {null_count} non-numeric values found.")
                except:
                    pass  # Non-numeric, skip
            elif df[column].dtype in ['int64', 'float64']:  # Check numeric columns
                # Skip .str.contains for numeric, just note potential issues
                if df[column].isna().sum() > 0 or df[column].dtype != df[column].infer_objects().dtype:
                    print(f"  {column} may need type validation (e.g., mixed types or NaN).")
        
        # 2. Outlier Detection (simple threshold: 3 standard deviations)
        print("\nOutlier Detection Check:")
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            mean = df[column].mean()
            std = df[column].std()
            outliers = df[(df[column] > mean + 3 * std) | (df[column] < mean - 3 * std)][column]
            if not outliers.empty:
                print(f"  {column} has {len(outliers)} potential outliers (e.g., values beyond 3 std dev).")
        
        # 3. Inconsistent Data Check (case and format variations)
        print("\nInconsistent Data Check:")
        for column in df.select_dtypes(include=['object']).columns:
            unique_values = df[column].dropna().unique()
            if len(unique_values) > 1 and df[column].str.lower().nunique() < len(unique_values):
                print(f"  {column} may have case inconsistencies (e.g., 'Sales' vs. 'sales').")
            # Check for date format variations (simple heuristic)
            if any(df[column].str.match(r'\d{4}-\d{2}-\d{2}', na=False)) and any(df[column].str.match(r'\w{3} \d{1,2}, \d{4}', na=False)):
                print(f"  {column} may have mixed date formats (e.g., '2025-07-05' vs. 'Jul 5, 2025').")
        
        # 4. Aggregation Feasibility
        print("\nAggregation Feasibility Check:")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print(f"  Can aggregate on {len(numeric_cols)} numeric columns (e.g., sum, average).")
            for col in numeric_cols:
                print(f"    {col}: Possible aggregations include sum, mean, etc.")
        else:
            print("  No numeric columns found for aggregation.")
            
    except FileNotFoundError:
        print(f"Error: {file} not found. Please check the file name or path.")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")