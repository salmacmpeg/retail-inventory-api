from httpx import AsyncClient


async def test_create_user_admin_success(client: AsyncClient):
    """Verifies an admin user is created successfully without a Customer Profile."""
    user_data = {
        "username": "New Admin",
        "password": "SecurePassword123!",
        "useremail": "new.admin@example.com",
        "userphone": "555-010-0002",
        "role": "admin",
    }
    response = await client.post("/user", json=user_data)
    assert response.status_code == 201

    data = response.json()
    assert data["role"] == "admin"
    assert data["customer_id"] is None  # Admin should not have a customer profile


async def test_login_success(client: AsyncClient):
    user_data = {
        "username": "New Admin",
        "password": "SecurePassword123!",
        "useremail": "new.admin@example.com",
        "userphone": "555-010-0002",
        "role": "admin",
    }
    response = await client.post("/user", json=user_data)
    assert response.status_code == 201

    data = response.json()
    assert data["role"] == "admin"
    assert data["customer_id"] is None  # Admin should not have a customer profile

    login_data = {
        "useremail": user_data["useremail"],
        "password": user_data["password"],  # Password defined inside fixture
    }
    response = await client.post("/login", json=login_data)
    assert response.status_code == 200

    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


async def test_login_invalid_credentials(client: AsyncClient, create_user_customer):
    user_data = {
        "username": "New Admin",
        "password": "SecurePassword123!",
        "useremail": "new.admin@example.com",
        "userphone": "555-010-0002",
        "role": "admin",
    }
    response = await client.post("/user", json=user_data)
    assert response.status_code == 201

    invalid_login = {"useremail": user_data["useremail"], "password": "WrongPasswordHere"}
    response = await client.post("/login", json=invalid_login)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


async def test_refresh_token_lifecycle(client: AsyncClient):
    user_data = {
        "username": "New Admins",
        "password": "SecurePassword123!s",
        "useremail": "new.admins@example.com",
        "userphone": "555-010-0002-12",
        "role": "admin",
    }
    response = await client.post("/user", json=user_data)
    assert response.status_code == 201

    # Step 1: Login to fetch initial refresh token
    login_data = {"useremail": user_data["useremail"], "password": user_data["password"]}
    login_resp = await client.post("/login", json=login_data)
    initial_tokens = login_resp.json()
    old_refresh_token = initial_tokens["refresh_token"]

    # Step 2: Use the valid refresh token to get a new pair
    refresh_payload = {"refresh_token": old_refresh_token}
    refresh_resp = await client.post("/refresh", json=refresh_payload)
    assert refresh_resp.status_code == 200

    new_tokens = refresh_resp.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens
    assert new_tokens["refresh_token"] != old_refresh_token

    # Step 3: Re-use old token again to make sure it was successfully revoked
    reused_resp = await client.post("/refresh", json=refresh_payload)
    assert reused_resp.status_code == 401
    assert reused_resp.json()["detail"] == "Invalid refresh token"


async def test_refresh_invalid_or_malformed_token(client: AsyncClient):
    """Verifies tampered token payloads yield HTTP 401."""
    bad_payload = {"refresh_token": "completely-malformed-jwt-token-string"}
    response = await client.post("/refresh", json=bad_payload)
    assert response.status_code == 401


async def test_logout_invalidates_token(client: AsyncClient, create_user_customer):
    """Validates logout sets is_revoked to true on the targeted refresh token record."""
    user_data = {
        "username": "New Admin",
        "password": "SecurePassword123!",
        "useremail": "new.admin@example.com",
        "userphone": "555-010-0002",
        "role": "admin",
    }
    response = await client.post("/user", json=user_data)
    assert response.status_code == 201

    # Login to fetch initial refresh token
    login_data = {"useremail": user_data["useremail"], "password": user_data["password"]}
    login_resp = await client.post("/login", json=login_data)
    tokens = login_resp.json()
    target_refresh_token = tokens["refresh_token"]

    # Step 2: Terminate session
    logout_payload = {"refresh_token": target_refresh_token}
    logout_resp = await client.post("/logout", json=logout_payload)
    assert logout_resp.status_code == 204
    assert logout_resp.content == b""  # HTTP 204 drops payload output entirely

    # Step 3: Confirm that the token is rejected by /refresh now that it's revoked
    refresh_resp = await client.post("/refresh", json=logout_payload)
    assert refresh_resp.status_code == 401
