from httpx import AsyncClient


async def test_create_customer(client: AsyncClient):
    # Test creating a new customer
    customer_data = {
        "username": "John Doe",
        "password": "Doepassword",
        "useremail": "john.Doe@example.com",
        "userphone": "123-456-7890-123",
        "role": "customer",
    }
    response = await client.post(url="/user", json=customer_data)
    response_data = response.json()

    assert response.status_code == 201, f"API Failed with body: {response.json()}"

    del customer_data["password"]  # Remove password from the data we want to check
    for key, value in customer_data.items():
        assert response_data[key] == value

    assert response_data["id"] == 1


async def test_create_customer_invalid_data(client: AsyncClient):
    # Missing 'email' and 'phone'
    bad_data = {"name": "Missing Info"}
    response = await client.post(url="/user", json=bad_data)
    assert response.status_code == 422  # Pydantic validation error


async def test_get_customer_exists(client: AsyncClient, authenticated_admin_token: str):
    # Test retrieving a customer by ID
    customer_data = {
        "username": "John Doe",
        "password": "Doepassword2",
        "useremail": "john.Doe@example.com",
        "userphone": "123-456-7890-123",
        "role": "customer",
    }
    response = await client.post(url="/user", json=customer_data)
    assert response.status_code == 201

    user_id = response.json()["id"]
    customer_id = response.json()["customer_id"]

    response = await client.get(url=f"/customer/{customer_id}", headers=authenticated_admin_token)
    response_data = response.json()

    assert response.status_code == 201, f"API Failed with body: {response.json()}"

    assert response_data["id"] == customer_id
    assert response_data["name"] == customer_data["username"]
    assert response_data["email"] == customer_data["useremail"]
    assert response_data["phone"] == customer_data["userphone"]
    assert response_data["user_id"] == user_id


async def test_get_nonexistent_customer(client: AsyncClient, authenticated_admin_token: str):
    # Test retrieving a non-existent customer
    response = await client.get(url="/customer/9999", headers=authenticated_admin_token)
    assert response.status_code == 404


async def test_get_customers(client: AsyncClient, authenticated_admin_token: str):
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

    # Test retrieving all customers
    response = await client.get(url="/customers", headers=authenticated_admin_token)
    assert response.status_code == 201
    response_data = response.json()
    assert len(response_data) >= len(customers)
