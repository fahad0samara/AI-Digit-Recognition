import pandas as pd
import numpy as np

def analyze_csv():
    # Read the CSV file in chunks
    chunk_size = 1000
    chunks = pd.read_csv('Test Data.csv', chunksize=chunk_size)
    
    # Initialize variables for analysis
    total_rows = 0
    column_names = None
    missing_values = {}
    data_types = None
    
    print("Analyzing data...")
    
    # Process each chunk
    for chunk_number, chunk in enumerate(chunks):
        if column_names is None:
            column_names = chunk.columns
            data_types = chunk.dtypes
            missing_values = {col: 0 for col in column_names}
            
        total_rows += len(chunk)
        
        # Count missing values
        for col in column_names:
            missing_values[col] += chunk[col].isnull().sum()
    
    # Print analysis results
    print("\nData Analysis Results:")
    print("-" * 50)
    print(f"Total number of rows: {total_rows}")
    print(f"\nColumns ({len(column_names)}):")
    for col, dtype in data_types.items():
        print(f"- {col} (Type: {dtype})")
        print(f"  Missing values: {missing_values[col]}")
        print(f"  Missing percentage: {(missing_values[col]/total_rows)*100:.2f}%")

if __name__ == "__main__":
    try:
        analyze_csv()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
