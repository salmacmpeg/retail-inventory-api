from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from RetailApp.core.config import config
from loguru import logger


def train_model(train_csv_path: str, model_version: str, model_output_path: str):
    """Trains a pipeline on cleaned data matching the application database structure."""
    # Load clean training data
    df = pd.read_csv(train_csv_path)

    X_train = df[["category", "inventory_level", "category_avg_price"]]
    y_train = df["optimal_price"]

    # Encoders built directly for your schemas.py data choices
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ("num", StandardScaler(), ["inventory_level", "category_avg_price"]),
        ]
    )

    # Bind structural steps and the regressor into a unified pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=500, random_state=42)),
        ]
    )

    # Fit model rules
    pipeline.fit(X_train, y_train)

   # VARIATION: Inject version into the directory path and auto-create the folder
    final_path = Path(model_output_path).parent / model_version / Path(model_output_path).name
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Export production binary artifact
    joblib.dump(pipeline, final_path)
    logger.info(f"Model production file compiled and saved to '{final_path}'")



if __name__ == "__main__":
    train_model(
        train_csv_path=config.ML_PRICER_TRAIN_DATA_PATH,
        model_version=config.ML_PRICER_MODEL_VERSION,
        model_output_path=config.ML_PRICER_MODEL_PATH,
    )
