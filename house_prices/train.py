import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def build_model():
    df = pd.read_csv("data/train.csv")

    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\t", "", regex=False)
    df.columns = df.columns.str.replace("\n", "", regex=False)

    target_col = [c for c in df.columns if "SalePrice" in c][0]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    joblib.dump(model, "models/model.pkl")

    return {"mse": mse}
