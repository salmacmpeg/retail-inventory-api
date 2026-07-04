from httpx import AsyncClient


async def test_create_user(client: AsyncClient):
    # Test creating a new user
    user_data = {
        "username": "John Doe",
        "password": "Doepassword",
        "useremail": "john.doe@example.com",
        "userphone": "123-456-7890",
        "role": "customer",
    }
    response = await client.post("/user", json=user_data)
    assert response.status_code == 201
    response_data = response.json()
    assert response_data["username"] == "John Doe"
    assert response_data["useremail"] == "john.doe@example.com"
    assert "password" not in response_data


async def test_create_user_existing_email(client: AsyncClient):
    # Test creating a user with an existing email
    user_data = {
        "username": "Jane Doe",
        "password": "Doepassword2",
        "useremail": "john.doe@example.com",  # Existing email
        "userphone": "098-765-4321",
        "role": "customer",
    }
    response = await client.post("/user", json=user_data)

    user_data = {
        "username": "Jane Doe",
        "password": "Doepassword2",
        "useremail": "john.doe@example.com",  # Existing email
        "userphone": "098-765-4321",
        "role": "customer",
    }
    response = await client.post("/user", json=user_data)
    assert response.status_code == 409  # Conflict


async def test_create_user_invalid_data(client: AsyncClient):
    # Test creating a user with invalid data
    user_data = {
        "username": "Invalid User",
        "password": "short",
        "userphone": "123",  # Invalid phone number
        "role": "customer",
    }
    response = await client.post("/user", json=user_data)
    assert response.status_code == 422  # Unprocessable Entity


async def test_get_my_data(client: AsyncClient, authenticated_customer_token: str):
    # Test retrieving current user data
    response = await client.get("/users/me", headers=authenticated_customer_token)
    assert response.status_code == 200, f"API Failed with body: {response.json()}"
    response_data = response.json()

    customer_data = {
        "username": "John Customer",
        "password": "customerpassword",
        "useremail": "john.customer@example.com",
        "userphone": "123-456-7890",
        "role": "customer",
    }
    assert response_data["username"] == customer_data["username"], f"API Failed with body: {response.json()}"
    assert response_data["useremail"] == customer_data["useremail"]
    assert response_data["userphone"] == customer_data["userphone"]
    assert response_data["role"] == customer_data["role"]
    assert response_data["id"] == 1
    assert response_data["is_active"] is True
    assert "password" not in response_data


async def test_get_my_data_unauthorized(client: AsyncClient):
    # Test retrieving current user data without authentication
    response = await client.get("/users/me")
    assert response.status_code == 401  # Unauthorized


async def test_delete_current_user(client: AsyncClient, authenticated_customer_token: str):
    # test delete un_utherized
    response = await client.delete("/users/me")
    assert response.status_code == 401  # unautherized

    # Test deleting the current user
    response = await client.delete("/users/me", headers=authenticated_customer_token)
    assert response.status_code == 204  # No Content

    # Verify the user is deleted
    response = await client.get("/users/me", headers=authenticated_customer_token)
    assert response.status_code == 401  # Unauthorized


async def test_list_all_users(client: AsyncClient, authenticated_admin_token: str):
    response = await client.get("/users/list")
    assert response.status_code == 401  # Unauthorized

    users_data = [
        {
            "username": "John Doe1",
            "password": "Doepassword1",
            "useremail": "john.doe1@example.com",
            "userphone": "123-456-7890-1",
            "role": "customer",
        },
        {
            "username": "John Doe2",
            "password": "Doepassword2",
            "useremail": "john.doe2@example.com",
            "userphone": "123-456-7890-2",
            "role": "admin",
        },
    ]
    for user in users_data:
        response = await client.post("/user", json=user)
        assert response.status_code == 201

    response = await client.get("/users/list", headers=authenticated_admin_token)
    assert response.status_code == 200
    response_data = response.json()
    for i, user in enumerate(response_data[1:]):
        assert user["username"] == users_data[i]["username"]


async def test_deactivate_user(client: AsyncClient, authenticated_admin_token: str):
    # first make the user
    user_data = {
        "username": "John Does",
        "password": "Doepasswords",
        "useremail": "john.does@example.com",
        "userphone": "123-456-78902",
        "role": "customer",
    }
    response = await client.post("/user", json=user_data)
    response_data = response.json()
    assert response.status_code == 201
    assert response_data["is_active"] is True

    user_id = response_data["id"]
    assert user_id is not None

    login_data = {"useremail": "john.does@example.com", "password": "Doepasswords"}
    response = await client.post("/login", json=login_data)
    tokens = response.json()
    header_customer = {"Authorization": f"Bearer {tokens['access_token']}"}

    response = await client.put(url=f"/users/{user_id}/deactivate")
    assert (
        response.status_code == 401
    ), f"API Failed with body: {response.json()}, and status code {response.status_code}"

    response = await client.put(url=f"/users/{user_id}/deactivate", headers=authenticated_admin_token)
    response_data = response.json()
    assert response_data["is_active"] is False, f"response body is {response_data}"

    response = await client.get("/users/me", headers=header_customer)
    assert (
        response.status_code == 403
    ), f"API Failed with body: {response.json()}, and status code {response.status_code}"

    # test activation
    response = await client.put(url=f"/users/{user_id}/activate")
    assert (
        response.status_code == 401
    ), f"API Failed with body: {response.json()}, and status code {response.status_code}"

    response = await client.put(url=f"/users/{user_id}/activate", headers=authenticated_admin_token)
    response_data = response.json()
    assert response_data["is_active"] is True, f"response body is {response_data}"

    response = await client.get("/users/me", headers=header_customer)
    assert (
        response.status_code == 200
    ), f"API Failed with body: {response.json()}, and status code {response.status_code}"


async def test_delete_current_user_success(client: AsyncClient):
    # 1. Create a customer user to test cascade deletion
    user_data = {
        "username": "DeleteMe",
        "password": "Password123!",
        "useremail": "delete.me@example.com",
        "userphone": "555-555-5555",
        "role": "customer",
    }
    create_response = await client.post("/user", json=user_data)
    assert create_response.status_code == 201

    # 2. Login to generate tokens and populate RefreshTokenTable
    login_data = {"useremail": "delete.me@example.com", "password": "Password123!"}
    login_response = await client.post("/login", json=login_data)
    assert login_response.status_code == 200

    tokens = login_response.json()
    user_headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    # 3. Verify Unauthorized requests are blocked
    unauth_response = await client.delete("/users/me")
    assert unauth_response.status_code == 401

    # 4. Delete the current user
    delete_response = await client.delete("/users/me", headers=user_headers)

    # Note: HTTP 204 No Content ignores response bodies.
    # FastAPI discards return dictionaries like {"detail": "..."} when status_code=204.
    assert delete_response.status_code == 204

    # 5. Verify data is gone by attempting to fetch profile
    # The dependency get_current_user will look up the ID, find nothing, and throw a 401/404
    get_profile_response = await client.get("/users/me", headers=user_headers)
    assert get_profile_response.status_code in [401, 404]
