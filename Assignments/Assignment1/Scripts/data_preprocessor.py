import pandas as pd
import numpy as np
from scipy import stats

def process(data):
    """
    Process the data by removing duplicates, fixing inconsistent entries,
    checking for outliers, and imputing missing values.
    """

    # Remove duplicates
    data = data.drop_duplicates()

    # Fix inconsistent entries in 'sex' column
    if 'sex' in data.columns:
        data['sex'] = data['sex'].astype(str).str.lower().apply(lambda x: 'female' if 'f' in x else 'male')

    # Detect and remove outliers using Z-score
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(data[numeric_cols]))
    data = data[~(z_scores > 3).any(axis=1)]

    # Identify missing values
    missing_data = data.isnull().sum()
    print("Missing values:\n", missing_data)

    # Impute missing values (optimized approach)
    data[numeric_cols] = data[numeric_cols].apply(lambda col: col.fillna(col.mean()))

    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Drop rows that still have missing values
    data.dropna(inplace=True)

    # Save the cleaned dataset
    data.to_csv("cleaned_data.csv", index=False)
    print("Cleaned data saved as 'cleaned_data.csv'")

    # Display final cleaned dataset preview
    print("Cleaned data preview:\n", data.head())

    return data