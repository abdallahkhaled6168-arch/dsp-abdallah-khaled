import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

def build_model():
    df = pd.read_csv("data/train.csv")

    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    X = X.drop(columns=["Id"], errors="ignore")

    X = X.fillna(0)

    X = pd.get_dummies(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")

    return {"mse": mse}
