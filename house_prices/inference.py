import pandas as pd
import joblib

def make_predictions():
    df = pd.read_csv("data/test.csv", sep="\t")
    df.columns = df.columns.str.strip()

    if "Id" in df.columns:
        ids = df["Id"]
        df = df.drop(columns=["Id"])
    else:
        ids = range(len(df))

    X = pd.get_dummies(df)
    X = X.fillna(0)

    model = joblib.load("models/model.pkl")

    preds = model.predict(X)

    output = pd.DataFrame({
        "Id": ids,
        "SalePrice": preds
    })

    output.to_csv("predictions.csv", index=False)

    print("Predictions file created successfully")
