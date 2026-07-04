import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from RetailApp.core.config import config
from RetailApp.database import get_db
from RetailApp.dependencies import get_current_admin
from RetailApp.models import User as UserTable
from RetailApp.schemas import PriceSuggestionRequest, PriceSuggestionResponse

router = APIRouter(prefix="/ml")


def load_ml_assets():
    base_path = Path(config.ML_PRICER_MODEL_PATH)
    versioned_dir = base_path.parent / config.ML_PRICER_MODEL_VERSION

    model_file_path = versioned_dir / base_path.name
    json_mapper_path = versioned_dir / "category_anchors.json"

    model = None
    category_anchors = {}
    if not json_mapper_path.exists():
        raise FileNotFoundError(f"Anchor map missing at: {json_mapper_path}")

    with open(json_mapper_path, "r", encoding="utf-8") as f:
        category_anchors = json.load(f)

    if json_mapper_path.exists():
        with open(json_mapper_path, "r", encoding="utf-8") as f:
            category_anchors = json.load(f)
    else:
        logger.warning(f"Category anchors not found at {json_mapper_path}")

    if not model_file_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_file_path}")

    # Load the pipeline model binary safely
    model = joblib.load(model_file_path)

    return model, category_anchors  


@router.post("/suggest_pricer", response_model=PriceSuggestionResponse, status_code=201)
async def add_order(
    request: Request,
    payload: PriceSuggestionRequest,
    db: AsyncSession = Depends(get_db),
    current_admin: UserTable = Depends(get_current_admin),
):
    
    model_pipeline = request.app.state.ml_model
    category_anchors = request.app.state.category_anchors

    if model_pipeline is None:
        return PriceSuggestionResponse(suggested_price=49.99)
    
    category_value = payload.category.value if hasattr(payload.category, "value") else payload.category
    category_avg_price = category_anchors.get(category_value, 45.00)

    try:
        # Build the exact features DataFrame your model expects
        # (Using 'inventory_level' to map back to the 'qty' dimension it was trained on)
        input_df = pd.DataFrame(
            [[payload.category, payload.inventory_level, category_avg_price]],
            columns=["category", "inventory_level", "category_avg_price"],
        )

        # Run model inference
        prediction = model_pipeline.predict(input_df)[0]
        suggested_price = round(float(prediction), 2)
        logger.info(f"Pricer mode suggested price {suggested_price} for payload {input_df}")
        return PriceSuggestionResponse(suggested_price=suggested_price)

    except Exception as e:
        logger.exception(f"ML inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"ML Engine Failure: {str(e)}")
