import pandas as pd
import joblib


def load_model(path="models/model.joblib"):
    return joblib.load(path)


def load_data(path: str):
    return pd.read_csv(path)


def make_predictions():
    model = load_model()
    df = load_data("data/test.csv")

    preds = model.predict(df)

    output = pd.DataFrame({
        "Id": df.index,
        "SalePrice": preds
    })

    output.to_csv("predictions.csv", index=False)
