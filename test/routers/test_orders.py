from httpx import AsyncClient


async def test_create_order_success(client: AsyncClient, authenticated_customer_token: str, sample_products):

    products = await sample_products([
        {"name": "iPhone", "category": "Electronics", "current_price": 1000.0, "inventory_level": 5},
        {"name": "Keyboard", "category": "Electronics", "current_price": 50.0, "inventory_level": 2}
    ])

    order_payload = {
        "status": "pending",
        "cart_items": [
            {"item_id": products[0]["id"], "quantity": 2},  # Cost: 2000.0
            {"item_id": products[1]["id"], "quantity": 1}   # Cost: 50.0
        ]
    }

    response = await client.post("/order", json=order_payload, headers=authenticated_customer_token)
    response_data = response.json()

    assert response.status_code == 201, f" failed api response with body: {response_data}"
    assert response_data["customer_id"] == 1
    assert response_data["order_total"] == 2050.0
    assert response_data["id"] == 1
    assert "created_at" in response_data
    assert response_data["status"] == "pending"
    assert len(response_data["cart_items"]) == 2


async def test_create_order_invalid_status(client: AsyncClient, authenticated_customer_token: str, sample_products):

    products = await sample_products([
        {"name": "iPhone", "category": "Electronics", "current_price": 1000.0, "inventory_level": 5},
        {"name": "Keyboard", "category": "Electronics", "current_price": 50.0, "inventory_level": 2}
    ])

    order_payload = {
        "status": "not_a_valid_status",
        "cart_items": [
            {"item_id": products[0]["id"], "quantity": 2},  # Cost: 2000.0
            {"item_id": products[1]["id"], "quantity": 1}   # Cost: 50.0
        ]
    }

    response = await client.post("/order", json=order_payload, headers=authenticated_customer_token)
    response_data = response.json()

    assert response.status_code == 422, f" failed api response with body: {response_data}"


async def test_get_empty_orders_valid_customer(client: AsyncClient, authenticated_customer_token: str):
    # Test getting empty orders for a valid customer
    response = await client.get(url="/orders/me", headers=authenticated_customer_token)
    assert response.status_code == 201, f"API Failed with body: {response.json()}"
    assert response.json() == []


async def test_get_order_by_id_authorized(client: AsyncClient, authenticated_customer_token: str, sample_products):

    # Test creating a valid order
    products = await sample_products([
        {"name": "iPhone", "category": "Electronics", "current_price": 1000.0, "inventory_level": 5},
        {"name": "Keyboard", "category": "Electronics", "current_price": 50.0, "inventory_level": 2}
    ])

    order_payload = {
        "status": "pending",
        "cart_items": [
            {"item_id": products[0]["id"], "quantity": 2},  # Cost: 2000.0
            {"item_id": products[1]["id"], "quantity": 1}   # Cost: 50.0
        ]
    }

    response = await client.post("/order", json=order_payload, headers=authenticated_customer_token)
    response_data = response.json()

    order_id = response_data["id"]

    # Test retrieval
    response = await client.get(f"/order/{order_id}", headers=authenticated_customer_token)
    assert response.status_code == 201, f" failed api response with body: {response_data}"
    assert response.json()["order_total"] == 2050.0
    assert response.json()["status"] == "pending"


async def test_get_order_by_id_un_authorized(client: AsyncClient, authenticated_customer_token: str):

    order_id = 9999

    # Test retrieval
    response = await client.get(f"/order/{order_id}", headers=authenticated_customer_token)
    assert response.status_code in (401, 403, 404), f" failed api response with body: {response.json()} and code : {response.status_code}"


async def test_create_order_insufficient_stock(client: AsyncClient, authenticated_customer_token: dict, sample_products):
    # 1. Arrange: Create a product with low stock levels
    products = await sample_products([
        {"name": "Limited Shoes", "category": "Cloths", "current_price": 100.0, "inventory_level": 1}
    ])
    
    # Requesting 2 when only 1 is available
    order_payload = {
        "status": "pending",
        "cart_items": [
            {"item_id": products[0]["id"], "quantity": 2}
        ]
    }

    # 2. Act: Execute request
    response = await client.post(
        "/order", 
        json=order_payload, 
        headers=authenticated_customer_token
    )

    # 3. Assert: Ensure your router catches the stock check exception
    assert response.status_code == 400
    assert "Not enough stock" in response.json()["detail"]


async def test_get_all_my_orders(client: AsyncClient, authenticated_customer_token: dict, sample_products):
    products = await sample_products([
        {"name": "Keyboard", "category": "Electronics", "current_price": 50.0, "inventory_level": 10}
    ])

    # Make 2 separate orders
    for _ in range(2):
        await client.post(
            "/order",
            json={"status": "pending", "cart_items": [{"item_id": products[0]["id"], "quantity": 1}]},
            headers=authenticated_customer_token
        )

    response = await client.get("/orders/me", headers=authenticated_customer_token)
    assert response.status_code == 201
    assert len(response.json()) == 2


async def test_get_all_customers_orders(client: AsyncClient, authenticated_customer_token: dict, authenticated_admin_token: dict, sample_products):
    products = await sample_products([
        {"name": "Keyboard", "category": "Electronics", "current_price": 50.0, "inventory_level": 10}
    ])

    # Place an order as a customer
    await client.post(
        "/order",
        json={"status": "pending", "cart_items": [{"item_id": products[0]["id"], "quantity": 1}]},
        headers=authenticated_customer_token
    )

    # Admin fetches global orders list
    response = await client.get("/orders", headers=authenticated_admin_token)
    assert response.status_code == 201
    assert len(response.json()) >= 1


# --- ADDITIONAL IMPORTANT EDGE CASE TESTS ---

async def test_create_order_product_not_found(client: AsyncClient, authenticated_customer_token: dict):
    order_payload = {
        "status": "pending",
        "cart_items": [
            {"item_id": 99999, "quantity": 1}  # ID does not exist
        ]
    }
    response = await client.post("/order", json=order_payload, headers=authenticated_customer_token)
    assert response.status_code == 404
    assert "doesnt exist" in response.json()["detail"]


async def test_create_order_invalid_quantity(client: AsyncClient, authenticated_customer_token: dict, sample_products):
    products = await sample_products([
        {"name": "Keyboard", "category": "Electronics", "current_price": 50.0, "inventory_level": 10}
    ])
    
    order_payload = {
        "status": "pending",
        "cart_items": [
            {"item_id": products[0]["id"], "quantity": 0}  # Triggers gt=0 Pydantic validation
        ]
    }
    response = await client.post("/order", json=order_payload, headers=authenticated_customer_token)
    assert response.status_code == 422


async def test_admin_update_order_status(client: AsyncClient, authenticated_customer_token: dict, authenticated_admin_token: dict, sample_products):
    products = await sample_products([
        {"name": "Keyboard", "category": "Electronics", "current_price": 50.0, "inventory_level": 10}
    ])

    create_resp = await client.post(
        "/order",
        json={"status": "pending", "cart_items": [{"item_id": products[0]["id"], "quantity": 1}]},
        headers=authenticated_customer_token
    )
    order_id = create_resp.json()["id"]

    # Act: Update status via patch endpoint as an admin
    patch_resp = await client.patch(
        f"/order/{order_id}/status",
        json={"new_status": "shipped"},
        headers=authenticated_admin_token
    )
    assert patch_resp.status_code == 201
    assert patch_resp.json()["status"] == "shipped"