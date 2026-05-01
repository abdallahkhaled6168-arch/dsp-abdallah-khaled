import pandas as pd


def preprocess_data(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    df = pd.get_dummies(df, columns=categorical_cols)

    return df
