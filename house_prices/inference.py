import pandas as pd
import joblib


def make_predictions():
    model = joblib.load("models/model.joblib")

    df = pd.read_csv("data/test.csv", sep="\s+")
    ids = df["Id"]

    preds = model.predict(df)

    output = pd.DataFrame({
        "Id": ids,
        "SalePrice": preds
    })

    output.to_csv("predictions.csv", index=False)

    return "Predictions saved"
