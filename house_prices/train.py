import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)


def save_model(model, path="models/model.joblib"):
    joblib.dump(model, path)


def build_model():
    df = load_data("data/train.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    mse = evaluate(model, X_test, y_test)
    save_model(model)
    return {"mse": mse}
