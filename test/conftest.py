import os

from loguru import logger

os.environ["ENV_STATE"] = "TEST"

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from RetailApp.core.config import config as app_config  # noqa: E402
from RetailApp.database import Base, get_db  # noqa: E402
from RetailApp.main import app  # noqa: E402


# This runs at the very start of the test session
def pytest_configure(config):

    logger.info(f"--- ACTIVE CONFIG: {type(app_config).__name__} ---")
    logger.info(f"--- DATABASE: {app_config.DATABASE_URL} ---")
    logger.info(f"--- FORCE ROLLBACK: {app_config.DB_FORCE_ROLL_BACK} ---")
    logger.info(f"--- LOG_LEVEL: {app_config.LOG_LEVEL} ---")


# fixture to make an async test database engine, create tables before tests
@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine(app_config.DATABASE_URL, echo=True, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


# fixture to make an async test database session, with optional rollback after each test
@pytest_asyncio.fixture
async def db_session(db_engine):
    async_session = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    async with async_session() as db_session:
        yield db_session
        if app_config.DB_FORCE_ROLL_BACK:
            await db_session.rollback()


# fixture to make a test client for the FastAPI app, using the test database session
@pytest_asyncio.fixture
async def client(db_session):

    # Override the get_db dependency to use the test db session
    # this is necessary because the get_db is a function, and must be overridden with a function
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    # remove the dependency override after the test
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def create_user_customer(client: AsyncClient):
    customer_data = {
        "username": "John Customer",
        "password": "customerpassword",
        "useremail": "john.customer@example.com",
        "userphone": "123-456-7890",
        "role": "customer",
    }
    response = await client.post("/user", json=customer_data)
    return response.json()


@pytest_asyncio.fixture
async def create_user_admin(client: AsyncClient):
    admin_data = {
        "username": "Admin User",
        "password": "adminpassword",
        "useremail": "admin@example.com",
        "userphone": "098-765-4321",
        "role": "admin",
    }
    response = await client.post("/user", json=admin_data)
    return response.json()


@pytest_asyncio.fixture
async def authenticated_customer_token(client: AsyncClient, create_user_customer):
    login_data = {"useremail": create_user_customer["useremail"], "password": "customerpassword"}
    logger.debug(f" customer id  is {create_user_customer['customer_id']}")
    response = await client.post("/login", json=login_data)
    tokens = response.json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest_asyncio.fixture
async def authenticated_admin_token(client: AsyncClient, create_user_admin):
    login_data = {"useremail": create_user_admin["useremail"], "password": "adminpassword"}
    response = await client.post("/login", json=login_data)
    tokens = response.json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}
