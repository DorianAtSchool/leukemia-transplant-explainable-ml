import pandas as pd
import pyreadstat
import sys
import os
from pathlib import Path

def read_sas7bdat_file(file_path, encoding='utf-8', **kwargs):
    """
    Read a SAS7BDAT file and return a pandas DataFrame
    
    Args:
        file_path (str): Path to the .sas7bdat file
        encoding (str): Character encoding (default: 'utf-8')
        **kwargs: Additional arguments for pyreadstat.read_sas7bdat()
    
    Returns:
        tuple: (DataFrame, metadata) where metadata contains file information
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        
        file_path = Path(file_path)
        if file_path.suffix.lower() != '.sas7bdat':
            print(f"Warning: File '{file_path}' doesn't have .sas7bdat extension.")
        
        # Read the SAS file
        df, meta = pyreadstat.read_sas7bdat(str(file_path), encoding=encoding, **kwargs)
        
        return df, meta
        
    except Exception as e:
        print(f"Error reading SAS file: {e}")
        return None, None

def display_sas_info(df, meta, max_rows=10, show_columns=True):
    """
    Display information about the SAS dataset
    
    Args:
        df (DataFrame): The pandas DataFrame
        meta: Metadata object from pyreadstat
        max_rows (int): Maximum number of rows to display
        show_columns (bool): Whether to show column information
    """
    if df is None:
        return
    
    print("=" * 60)
    print("SAS7BDAT FILE INFORMATION")
    print("=" * 60)
    
    # Basic info
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"File encoding: {meta.file_encoding if hasattr(meta, 'file_encoding') else 'Unknown'}")
    print(f"Creation date: {meta.creation_time if hasattr(meta, 'creation_time') else 'Unknown'}")
    print(f"Modified date: {meta.modification_time if hasattr(meta, 'modification_time') else 'Unknown'}")
    
    if hasattr(meta, 'table_name') and meta.table_name:
        print(f"Table name: {meta.table_name}")
    
    # Column information
    if show_columns and df.shape[1] > 0:
        print("\n" + "=" * 60)
        print("COLUMN INFORMATION")
        print("=" * 60)
        
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            
            # Get column label if available
            label = ""
            if hasattr(meta, 'column_labels') and meta.column_labels and col in meta.column_labels:
                label = f" (Label: {meta.column_labels[col]})"
            
            print(f"{i+1:2d}. {col:<20} {str(dtype):<10} Non-null: {non_null:<8} Null: {null_count}{label}")
    
    # Data preview
    print("\n" + "=" * 60)
    print(f"DATA PREVIEW (First {min(max_rows, len(df))} rows)")
    print("=" * 60)
    print(df.head(max_rows).to_string())
    
    # Summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print("\n" + "=" * 60)
        print("NUMERIC COLUMNS SUMMARY")
        print("=" * 60)
        print(df[numeric_cols].describe().to_string())

def save_to_formats(df, base_name, formats=['csv', 'xlsx', 'parquet']):
    """
    Save the DataFrame to various formats
    
    Args:
        df (DataFrame): The pandas DataFrame
        base_name (str): Base filename without extension
        formats (list): List of formats to save to
    """
    if df is None:
        return
    
    saved_files = []
    
    for fmt in formats:
        try:
            if fmt == 'csv':
                filename = f"{base_name}.csv"
                df.to_csv(filename, index=False)
                saved_files.append(filename)
            elif fmt == 'xlsx':
                filename = f"{base_name}.xlsx"
                df.to_excel(filename, index=False)
                saved_files.append(filename)
            elif fmt == 'parquet':
                filename = f"{base_name}.parquet"
                df.to_parquet(filename, index=False)
                saved_files.append(filename)
            elif fmt == 'json':
                filename = f"{base_name}.json"
                df.to_json(filename, orient='records', indent=2)
                saved_files.append(filename)
        except Exception as e:
            print(f"Error saving to {fmt}: {e}")
    
    if saved_files:
        print(f"\nSaved to: {', '.join(saved_files)}")

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        # Interactive mode
        file_path = input("Enter path to .sas7bdat file: ").strip()
        if not file_path:
            print("No file path provided.")
            return
    else:
        file_path = sys.argv[1]
    
    # Optional parameters
    encoding = 'utf-8'
    if len(sys.argv) > 2:
        encoding = sys.argv[2]
    
    max_rows = 10
    if len(sys.argv) > 3:
        try:
            max_rows = int(sys.argv[3])
        except ValueError:
            print("Invalid max_rows value, using default: 10")
    
    # Read the file
    print(f"Reading SAS file: {file_path}")
    print(f"Using encoding: {encoding}")
    
    df, meta = read_sas7bdat_file(file_path, encoding=encoding)
    
    if df is not None:
        # Display information
        display_sas_info(df, meta, max_rows=max_rows)
        
        # Ask if user wants to save to other formats
        if len(sys.argv) < 2:  # Interactive mode
            save_choice = input("\nSave to other formats? (y/n): ").strip().lower()
            if save_choice == 'y':
                base_name = Path(file_path).stem
                formats = input("Enter formats (csv,xlsx,parquet,json) [csv]: ").strip()
                if not formats:
                    formats = 'csv'
                format_list = [f.strip() for f in formats.split(',')]
                save_to_formats(df, base_name, format_list)

# Example usage functions
def example_read_basic():
    """Example: Basic reading of a SAS file"""
    df, meta = read_sas7bdat_file('example.sas7bdat')
    if df is not None:
        print(df.head())
        print(f"Shape: {df.shape}")

def example_read_with_options():
    """Example: Reading with specific options"""
    df, meta = read_sas7bdat_file(
        'example.sas7bdat',
        encoding='latin1',  # Different encoding
        # usecols=['var1', 'var2'],  # Read only specific columns
        # chunksize=1000  # Read in chunks for large files
    )
    return df, meta

if __name__ == "__main__":
    main()
