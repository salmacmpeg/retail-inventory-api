import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from RetailApp.core.config import config
from loguru import logger


def evaluate_model(test_csv_path: str, model_version: str, model_path: str, metrics_base_path: str):
    """Loads the model artifact to test prediction precision."""
    # Load model and test slice

    final_path = Path(model_path).parent / model_version / Path(model_path).name

    logger.info(f"Loading model artifact from: '{final_path}'")
    
    model = joblib.load(final_path)

    df = pd.read_csv(test_csv_path)

    X_test = df[["category", "inventory_level", "category_avg_price"]]
    y_test = df["optimal_price"]

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate model precision metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    logger.debug("--- Model Performance Metrics ---")
    logger.debug(f"Mean Absolute Error (MAE): ${mae:.2f}")
    logger.debug(f"R-squared Score (Variance Explained): {r2 * 100:.2f}%")

    metrics_path_obj = Path(metrics_base_path)
    final_metrics_path = metrics_path_obj.parent / f"{metrics_path_obj.stem}_{model_version}{metrics_path_obj.suffix}"
    
    # Ensure target folder structures exist before exporting
    final_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "model_version": model_version,
        "mean_absolute_error": round(float(mae), 4),
        "r2_score": round(float(r2), 4),
        "test_dataset_size": len(df)
    }

    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
        
    logger.info(f"Performance report tracking records exported safely to: '{final_metrics_path}'")


if __name__ == "__main__":
    evaluate_model(
        test_csv_path=config.ML_PRICER_TEST_DATA_PATH, 
        model_version=config.ML_PRICER_MODEL_VERSION,
        model_path=config.ML_PRICER_MODEL_PATH,
        metrics_base_path=config.ML_PRICER_METRICS_PATH
    )
