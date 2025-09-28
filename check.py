#### script to check NULL and duplicates in the datasets selected 


import pandas as pd

# List of CSV files to check (replace with your actual file names)
files = [
    "WA_Fn-UseC_-HR-Employee-Attrition.csv",           # IBM HR Analytics Attrition Dataset
    "supply_chain_data.csv", # Supply Chain Analysis
    "smart_logistics_dataset.csv"     # Smart Logistics Supply Chain Dataset
]

# Loop through each file
for file in files:
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        print(f"\nChecking {file}:")
        if duplicate_rows > 0:
            print(f"Found {duplicate_rows} duplicate rows.")
        else:
            print("No duplicate rows found.")
        
        # Check for duplicates in ID columns (assuming common ID fields)
        id_columns = [col for col in df.columns if "id" in col.lower() or "number" in col.lower()]
        for col in id_columns:
            duplicates_in_col = df[col].duplicated().sum()
            if duplicates_in_col > 0:
                print(f"Found {duplicates_in_col} duplicates in {col} column.")
        
        # Check for null values (NaN or blank) in each column
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print("Null values found in the following columns:")
            for column, count in null_counts.items():
                if count > 0:
                    print(f"  {column}: {count} null values")
        else:
            print("No null values found.")
            
    except FileNotFoundError:
        print(f"Error: {file} not found. Please check the file name or path.")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")