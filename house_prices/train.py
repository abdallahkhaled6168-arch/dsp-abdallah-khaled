import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
from house_prices.preprocess import preprocess_data

def build_model():
    df = pd.read_csv("data/train.csv", sep="\s+")

    df = preprocess_data(df)

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    joblib.dump(model, "models/model.joblib")

    return {"mae": mae}
