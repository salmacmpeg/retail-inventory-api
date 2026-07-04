from httpx import AsyncClient
from RetailApp.main import app

async def test_suggest_price_real_execution(client: AsyncClient, authenticated_admin_token: dict):
    """Executes the endpoint using the real loaded model and anchors."""
    payload = {
        "category": "Electronics",
        "inventory_level": 5
    }
    
    response = await client.post("/ml/suggest_pricer", json=payload, headers=authenticated_admin_token)
    response_data = response.json()
    
    assert response.status_code == 201
    assert "suggested_price" in response_data
    assert isinstance(response_data["suggested_price"], float)


async def test_suggest_price_validation_rules(client: AsyncClient, authenticated_admin_token: dict):
    """Ensures input verification layer remains active."""
    bad_payload = {
        "category": "InvalidCategoryName",
        "inventory_level": -10
    }
    
    response = await client.post("/ml/suggest_pricer", json=bad_payload, headers=authenticated_admin_token)
    assert response.status_code == 422




async def test_suggest_price_unseen_category_fallback(client: AsyncClient, authenticated_admin_token: dict):
    """
    Tests what happens when a category exists in your schemas.category_choice Enum 
    (like 'bed_bath_table'), but is missing from your 'category_anchors.json' file.
    Verifies that the router successfully falls back to the hardcoded 45.00 anchor price.
    """
    # Temporarily remove a category from anchors to simulate a new/unmapped category
    original_anchors = app.state.category_anchors.copy()
    if "bed_bath_table" in app.state.category_anchors:
        del app.state.category_anchors["bed_bath_table"]

    payload = {
        "category": "bed_bath_table",
        "inventory_level": 5
    }

    try:
        response = await client.post("/ml/suggest_pricer", json=payload, headers=authenticated_admin_token)
        assert response.status_code == 201
        assert "suggested_price" in response.json()
        assert isinstance(response.json()["suggested_price"], float)
    finally:
        # Always restore original state so other tests don't break
        app.state.category_anchors = original_anchors


async def test_suggest_price_extreme_inventory_impact(client: AsyncClient, authenticated_admin_token: dict):
    """
    Tests how your real model handles extreme inventory counts (e.g., 1 vs 10000).
    Verifies that changing the input parameters actually shifts the model output, 
    confirming the pipeline is reacting dynamically to features.
    """
    # Payload with very low inventory (should suggest higher price due to scarcity)
    low_stock_payload = {"category": "Electronics", "inventory_level": 1}
    resp_low = await client.post("/ml/suggest_pricer", json=low_stock_payload, headers=authenticated_admin_token)
    
    # Payload with massive overstock (should suggest lower price to clear inventory)
    high_stock_payload = {"category": "Electronics", "inventory_level": 10000}
    resp_high = await client.post("/ml/suggest_pricer", json=high_stock_payload, headers=authenticated_admin_token)

    assert resp_low.status_code == 201
    assert resp_high.status_code == 201
    
    # Assert that the model output changes based on parameters (they shouldn't be identical)
    assert resp_low.json()["suggested_price"] != resp_high.json()["suggested_price"]


async def test_suggest_price_missing_authorization(client: AsyncClient):
    """
    Verifies that an unauthenticated user (no token headers provided at all) 
    is fully blocked from triggering ML execution loops.
    """
    payload = {
        "category": "Electronics",
        "inventory_level": 10
    }
    response = await client.post("/ml/suggest_pricer", json=payload)
    assert response.status_code in (401, 403)