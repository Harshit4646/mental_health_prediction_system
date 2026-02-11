import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_data.csv")
    df = encode_features(df)
    df.to_csv("data/processed/final_data.csv", index=False)
