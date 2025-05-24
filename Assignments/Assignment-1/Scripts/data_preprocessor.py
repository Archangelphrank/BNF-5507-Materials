import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        pass  # No arguments needed during initialization

    def remove_duplicates(self, df):
        """Removes duplicate rows."""
        return df.drop_duplicates()

    def fix_format(self, df):
        """Standardizes categorical entries while preserving male and female distinctions."""
        df['sex'] = df['sex'].astype(str).str.strip().str.lower()

        # Mapping common variations to 'male' or 'female'
        df['sex'] = df['sex'].replace({
            'f': 'female', 'femalee': 'female', 'fem': 'female', 'woman': 'female',
            'm': 'male', 'malee': 'male', 'mal': 'male', 'man': 'male'
        })
        return df

    def handle_outliers(self, df):
        """Identifies and removes outliers using Z-score."""
        outliers = np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) > 3
        df.loc[outliers.any(axis=1), 'tprc'] = np.mean(df['tprc'])
        return df

    def handle_missing_values(self, df):
        """Imputes missing values."""
        df['age'].fillna(df['age'].mean(), inplace=True)
        df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
        df.dropna(inplace=True)
        return df

    def encode_categorical(self, df):
        """Encodes categorical variables using Label Encoding."""
        le = LabelEncoder()
        df['sex'] = le.fit_transform(df['sex'])
        df['embarked'] = le.fit_transform(df['embarked'])
        return df

    def transform(self, df):
        """Applies all preprocessing steps sequentially."""
        df = self.remove_duplicates(df)
        df = self.fix_format(df)
        df = self.handle_outliers(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)
        return df