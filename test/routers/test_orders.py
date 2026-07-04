from httpx import AsyncClient


async def test_create_order_success(client: AsyncClient, authenticated_customer_token: str):

    # Test creating a valid order
    order_data = {"order_total": 99.99, "status": "pending"}
    response = await client.post("/order", json=order_data, headers=authenticated_customer_token)
    response_data = response.json()

    assert response.status_code == 201, f" failed api response with body: {response_data}"
    assert response_data["customer_id"] == 1
    assert response_data["order_total"] == 99.99
    assert response_data["id"] == 1
    assert "created_at" in response_data


async def test_create_order_invalid_status(client: AsyncClient, authenticated_customer_token: str):

    invalid_order = {
        "order_total": 10,
        "status": "not_a_valid_status",  # Not in Choice enum
    }
    response = await client.post("/order", json=invalid_order, headers=authenticated_customer_token)
    assert response.status_code == 422  # Unprocessable Entity


async def test_get_empty_orders_valid_customer(client: AsyncClient, authenticated_customer_token: str):
    # Test getting empty orders for a valid customer
    response = await client.get(url="/orders/me", headers=authenticated_customer_token)
    assert response.status_code == 201, f"API Failed with body: {response.json()}"
    assert response.json() == []


async def test_get_order_by_id_authorized(client: AsyncClient, authenticated_customer_token: str):

    # Test creating a valid order
    order_data = {"order_total": 99.99, "status": "pending"}
    response = await client.post("/order", json=order_data, headers=authenticated_customer_token)
    response_data = response.json()

    order_id = response_data["id"]

    # Test retrieval
    response = await client.get(f"/order/{order_id}", headers=authenticated_customer_token)
    assert response.status_code == 201, f" failed api response with body: {response_data}"
    assert response.json()["order_total"] == 99.99
    assert response.json()["status"] == "pending"


async def test_get_order_by_id_un_authorized(client: AsyncClient, authenticated_customer_token: str):

    # Test creating a valid order
    order_data = {"order_total": 99.99, "status": "pending"}
    response = await client.post("/order", json=order_data, headers=authenticated_customer_token)
    response_data = response.json()

    order_id = response_data["id"]

    # Test retrieval
    response = await client.get(f"/order/{order_id}")
    assert (
        response.status_code == 401
    ), f" failed api response with body: {response.json()} and code : {response.status_code}"


async def test_get_nonexistent_order(client: AsyncClient, authenticated_customer_token: str):
    response = await client.get("/order/8888", headers=authenticated_customer_token)
    assert response.status_code == 404


async def test_get_all_my_orders(client: AsyncClient, authenticated_customer_token: str):
    order_data = [{"order_total": 201, "status": "pending"}, {"order_total": 150, "status": "shipped"}]

    for order in order_data:
        await client.post(url="/order", json=order, headers=authenticated_customer_token)

    response = await client.get(url="/orders/me", headers=authenticated_customer_token)
    assert response.status_code == 201, f"API Failed with body: {response.json()}"

    response_data = response.json()

    assert len(response_data) == 2

    for i, order in enumerate(response_data):
        assert response_data[i]["customer_id"] == 1
        assert response_data[i]["order_total"] == order_data[i]["order_total"]
        assert response_data[i]["status"] == order_data[i]["status"]
        assert "created_at" in response_data[i]
        assert "id" in response_data[i]


async def test_get_all_customers_orders(client: AsyncClient, authenticated_admin_token: str):
    # Create multiple customers
    customers = [
        {
            "username": "John Doe1",
            "password": "Doepassword1",
            "useremail": "john.Doe1@example.com",
            "userphone": "123-456-7890-123",
            "role": "customer",
        },
        {
            "username": "John Doe2",
            "password": "Doepassword2",
            "useremail": "john.Doe2@example.com",
            "userphone": "223-456-7890-123",
            "role": "customer",
        },
        {
            "username": "John Doe3",
            "password": "Doepassword3",
            "useremail": "john.Doe3@example.com",
            "userphone": "323-456-7890-123",
            "role": "customer",
        },
    ]
    for customer_data in customers:
        await client.post(url="/user", json=customer_data)

    # login first user
    for i in range(len(customers)):
        login_data = {"useremail": customers[i]["useremail"], "password": customers[i]["password"]}
        response = await client.post("/login", json=login_data)
        tokens = response.json()
        header_customer = {"Authorization": f"Bearer {tokens['access_token']}"}

        # place orders for this customer
        order_data = [{"order_total": 201, "status": "pending"}, {"order_total": 150, "status": "shipped"}]
        for order in order_data:
            await client.post(url="/order", json=order, headers=header_customer)

    # Test retrieving all customers
    response = await client.get(url="/orders", headers=authenticated_admin_token)
    assert response.status_code == 201
    response_data = response.json()
    assert len(response_data) == len(customers) * 2
