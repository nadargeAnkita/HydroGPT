import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ROBUST PATH HANDLING


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")

def validate():

    print("=== Loading merged_cleaned_dataset.csv ===")

    file_path = os.path.join(DATA_DIR, "merged_cleaned_dataset.csv")

    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.set_index("date")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    print("✔ Dataset loaded successfully!")
    print("Shape:", df.shape)

   
    # Missing Values
   

    print("\n=== Missing Values ===")
    print(df.isna().sum())

   
    # Duplicate Date Check
   
    if df.index.duplicated().sum() > 0:
        print("\n⚠ Duplicate timestamps found!")
    else:
        print("\n✔ No duplicate timestamps.")

  
    # Continuous Date Check
   

    expected_range = pd.date_range(start=df.index.min(),
                                   end=df.index.max(),
                                   freq='D')

    missing_dates = expected_range.difference(df.index)

    print(f"\nMissing Dates Count: {len(missing_dates)}")

  
    # Outlier Check using IQR
   

    print("\n=== Outlier Check using IQR ===")

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    for col in df.columns:
        outliers = ((df[col] < (Q1[col] - 1.5 * IQR[col])) |
                    (df[col] > (Q3[col] + 1.5 * IQR[col]))).sum()
        print(f"{col}: {outliers}")

    
    # Target Statistics
   

    print("\n=== Target Statistics (inflow_cumecs) ===")
    print(df["inflow_cumecs"].describe())

  
    # Zero Variance Check
  

    zero_var_cols = df.columns[df.nunique() <= 1]
    if len(zero_var_cols) > 0:
        print("\n⚠ Zero variance columns:", list(zero_var_cols))
    else:
        print("\n✔ No zero variance columns.")

    
    # Correlation with Target
   

    print("\n=== Correlation with Target ===")
    corr_target = df.corr()["inflow_cumecs"].sort_values(ascending=False)
    print(corr_target)

  
    # Date Range
    

    print("\n=== Date Range ===")
    print("Start Date:", df.index.min())
    print("End Date:", df.index.max())

   
    # Years Present
    

    if df.empty or df.index.isna().all():
        print("\nYears present in dataset: []")
    else:
        years = sorted(df.index.year.dropna().unique())
        print("\nYears present in dataset:", years)

    print("\nValidation Completed Successfully!")


if __name__ == "__main__":
    validate()