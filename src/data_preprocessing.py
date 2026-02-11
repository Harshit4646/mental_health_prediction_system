import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['Timestamp', 'comments'], errors='ignore')

    # Standardize text
    df.columns = df.columns.str.lower()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Fix age outliers
    df = df[(df['age'] >= 18) & (df['age'] <= 70)]

    return df

if __name__ == "__main__":
    df = load_data("data/raw/survey.csv")
    df = clean_data(df)
    df.to_csv("data/processed/cleaned_data.csv", index=False)
