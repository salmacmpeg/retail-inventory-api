from httpx import AsyncClient

from test.conftest import authenticated_admin_token, authenticated_customer_token

async def test_create_product_success(client: AsyncClient, authenticated_admin_token: str):
    # Test creating a valid product
    product_data = {
        "name": "Test Product",
        "category": "Cloths",
        "cost_price": 19.99,
        "current_price": 20.5,
        "inventory_level": 100,
    }
    response = await client.post("/product", json=product_data, headers=authenticated_admin_token)
    response_data = response.json()

    assert response.status_code == 201, f"API Failed with body: {response.json()}"
    assert response_data["name"] == product_data["name"]
    assert response_data["category"] == product_data["category"]
    assert response_data["cost_price"] == product_data["cost_price"]
    assert response_data["current_price"] == product_data["current_price"]
    assert response_data["inventory_level"] == product_data["inventory_level"]
    assert "id" in response_data

async def test_create_product_invalid_data(client: AsyncClient, authenticated_admin_token: str):
    # Missing 'name' and 'category'
    bad_data = {"cost_price": 19.99, "current_price": 20.5, "inventory_level": 100}
    response = await client.post("/product", json=bad_data, headers=authenticated_admin_token)
    assert response.status_code == 422  # Pydantic validation error

    # Missing 'cost_price' and 'current_price' and 'inventory_level'
    bad_data = {"name": "Test Product", "category": "Test Category"}
    response = await client.post("/product", json=bad_data, headers=authenticated_admin_token)
    assert response.status_code == 422


async def test_get_all_products(client: AsyncClient, authenticated_admin_token: str):
    # Test retrieving all products
    product_data = {
        "name": "Test Product",
        "category": "Cloths",
        "cost_price": 19.99,
        "current_price": 20.5,
        "inventory_level": 100,
    }
    response = await client.post("/product", json=product_data, headers=authenticated_admin_token)
    assert response.status_code == 201, f"API Failed with body: {response.json()}"

    response = await client.get("/products", headers=authenticated_admin_token)
    assert response.status_code == 201, f"API Failed with body: {response.json()}"
    assert len(response.json()) >= 1  # At least one product should be present

    # Test retrieving a specific product
    product_id = 1
    response = await client.get(f"/product/{product_id}", headers=authenticated_admin_token)
    assert response.status_code == 201, f"API Failed with body: {response.json()}"

    # Test retrieving a non-existent product
    non_existent_product_id = 9999
    response = await client.get(f"/product/{non_existent_product_id}", headers=authenticated_admin_token)
    assert response.status_code == 404