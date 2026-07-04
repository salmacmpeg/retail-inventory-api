from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from RetailApp.core.config import config
from loguru import logger


def preprocess_data(raw_csv_path: str, train_out: str, test_out: str):
    """Cleans Kaggle data and splits it to preserve schema alignment."""
    # Load raw dataset
    df = pd.read_csv(raw_csv_path)

    logger.info(f"Loading Raw Data from {raw_csv_path} with size {df.shape}")

    category_baselines = df.groupby('product_category_name')['unit_price'].mean().to_dict()
    
    mapping_path = Path(config.ML_PRICER_MODEL_PATH).parent / config.ML_PRICER_MODEL_VERSION / "category_anchors.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(category_baselines, f, indent=4)
    logger.info(f"Saved category baseline anchors map to {mapping_path}")
   
    # Map raw columns to your exact schema definitions
    processed_df = pd.DataFrame(
        {
            "category": df["product_category_name"],
            "inventory_level": df["qty"],  # demand/volume signal
            "category_avg_price": df["product_category_name"].map(category_baselines),
            "optimal_price": df["unit_price"],  # Target variable
        }
    )

    # Handle missing data if any exist
    processed_df = processed_df.dropna()

    logger.info(f"Processed Data size is {processed_df.shape}")

    # Split into separate datasets (80% train, 20% test)
    train_df, test_df = train_test_split(
        processed_df, test_size=0.2, random_state=42
    )

    logger.info(f"Splitting Data into Train {train_df.shape} and Test {test_df.shape}")

    # Save to disk as clean CSVs
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info(f"Data split successfully! Saved to '{train_out}' and '{test_out}'")


if __name__ == "__main__":
    preprocess_data(
        raw_csv_path=config.ML_PRICER_RAW_DATA_PATH,
        train_out=config.ML_PRICER_TRAIN_DATA_PATH,
        test_out=config.ML_PRICER_TEST_DATA_PATH,
    )
