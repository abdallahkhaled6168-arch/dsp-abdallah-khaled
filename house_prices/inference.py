import pandas as pd
import joblib

def make_predictions():
    model = joblib.load("models/model.pkl")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    y = train_df["SalePrice"]
    train_df = train_df.drop(columns=["SalePrice"])

    train_df = train_df.drop(columns=["Id"], errors="ignore")
    test_ids = test_df["Id"]
    test_df = test_df.drop(columns=["Id"], errors="ignore")

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)

    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    preds = model.predict(test_df)

    output = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": preds
    })

    output.to_csv("predictions.csv", index=False)
