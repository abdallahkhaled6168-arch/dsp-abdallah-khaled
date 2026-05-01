import pandas as pd
import joblib
from house_prices.preprocess import preprocess_data

def make_predictions():
    model = joblib.load("models/model.joblib")

    df = pd.read_csv("data/test.csv", sep="\s+")
    ids = df["Id"]

    df = preprocess_data(df)

    if "SalePrice" in df.columns:
        df = df.drop("SalePrice", axis=1)

    predictions = model.predict(df)

    output = pd.DataFrame({
        "Id": ids,
        "SalePrice": predictions
    })

    output.to_csv("predictions.csv", index=False)

    return "Predictions saved!"
