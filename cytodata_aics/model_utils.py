from pathlib import Path
import pandas as pd

def save_predictions_classifier(preds, output_dir):
    """
    TODO: make this better? maybe use vol predictor code?
    """
    records = []
    for pred in preds:
        record = dict()
        for col in ["id", "y", "yhat"]:
            record[col] = pred[col].squeeze().numpy()
        record["loss"] = [pred["loss"].item()] * len(pred["id"])
        records.append(record)

    pd.DataFrame.from_records(records).to_csv(
        Path(output_dir) / "model_predictions.csv"
    )