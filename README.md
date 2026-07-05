# Retail Inventory

This project is a small retail inventory system built around FastAPI. It helps a shop manage products, keep track of stock, handle customer orders, and get price suggestions from a simple machine learning model.

It is not a large enterprise platform. It is a practical backend with a lightweight frontend, aimed at showing how a real retail workflow can be organized in a clean and testable way.

## life URL: 
https://retail-inventory-api-production.up.railway.app

## What the system does

The app lets you:

- create and manage products with name, category, cost price, selling price, and inventory level
- register users and separate customer and admin roles
- authenticate users with JWT-based login and refresh tokens
- place orders, reduce inventory, and view order history
- update order status from the admin side
- ask for a price suggestion based on product category and current stock level

The frontend pages live under the frontend folder, while the API is served by the FastAPI application in RetailApp.

## Demo

![Retail Inventory Demo](./assets/Animation.gif)

## Architecture

The system runs as two Docker containers managed by Docker Compose:

- **app**: FastAPI application serving the API and frontend static files on port 8000
- **db**: MySQL 8.0 database with a persistent named volume for data storage

The app container waits for the database to pass a health check before starting,
ensuring reliable startup order. Configuration is fully environment-based with
separate Dev, Prod, and Test profiles.
```text
┌───────────────────────────────────────────┐
│            docker-compose                 │
│                                           │
│  ┌──────────────┐      ┌───────────────┐  │
│  │   app        │      │   db          │  │
│  │   FastAPI    │───▶ │   MySQL 8.0   │  │
│  │   port 8000  │      │   port 3306   │  │
│  └──────┬───────┘      └───────┬───────┘  │
│         │                    │            │
└─────────┼────────────────────┼────────────┘
          │                    │
    browser/client        mysql_data
    localhost:8000        (named volume)
```

Request flow:
Browser → FastAPI router → SQLAlchemy async session → MySQL

## Main technologies

The project uses a mix of backend, data, and deployment tools:

- FastAPI for the API layer and routing
- SQLAlchemy with async support for database access
- Pydantic for request and response validation
- JWT authentication with jose and password hashing with passlib
- MySQL in production-style setup, with SQLite support for tests
- Docker and Docker Compose for running the app and database together
- Uvicorn as the ASGI server
- Loguru for structured application logging
- pytest for automated tests
- scikit-learn, pandas, numpy, and joblib for the pricing model
- plain HTML, CSS, and JavaScript for the simple frontend pages

## Project structure

- RetailApp/main.py: application entry point and router registration
- RetailApp/routes: API endpoints for authentication, customers, orders, products, users, and pricing
- RetailApp/models.py and RetailApp/schemas.py: database models and request/response schemas
- RetailApp/core: configuration, logging, and security helpers
- PricerMlModel: preprocessing, training, and inference assets for the pricing module
- frontend: simple pages for login, registration, admin, and customer views
- test: router-level tests covering auth, customers, orders, and users

## Best practices used here

This repo is structured in a way that keeps the code understandable and easier to extend:

- clear separation of concerns between routes, schemas, models, and configuration
- dependency injection for database sessions and authenticated users
- input validation through Pydantic so bad requests are rejected early
- async database access for better scalability in API requests
- role-based access so admins and customers see different actions
- environment-based configuration instead of hardcoded secrets
- request logging with request IDs to make debugging simpler
- automated tests around the main API routes
- containerization so the backend and database can be run consistently
- retry logic on database connection at startup to handle container timing gracefully
- health-checked database container so the app only starts when MySQL is fully ready

## How to run the project

### Option 1: with Docker

This is the easiest way to run everything together.

```bash
docker-compose up --build
```

Once the containers are up, open:

- API docs: http://localhost:8000/docs
- frontend entry page: http://localhost:8000/

### Option 2: locally with Python

If you want to run it directly on your machine, first install the dependencies:

```bash
pip install -r requirements.txt
```

Then start the app:

```bash
uvicorn RetailApp.main:app --host 127.0.0.1 --port 8000 --reload
```

The Makefile also includes useful shortcuts:

```bash
make install
make run
make test
make lint
```

## How to use the system

### 1. Create an account

Use the registration endpoint or the UI to create a user. The role can be set to customer, employee, warehouse_manager, or admin.

For admin actions such as adding inventory or requesting pricing suggestions, create a user with the admin role.

### 2. Sign in

Use the login endpoint to get an access token and a refresh token. The API docs at /docs are the easiest place to try this interactively.

### 3. Manage products

Admins can add products and define their:

- name
- category
- cost price
- current price
- inventory level

Products are validated before being stored, and inventory must be greater than zero when created.

### 4. Place orders

Customers can create orders by sending a list of items and quantities. The API checks that:

- the product exists
- enough stock is available
- the order can be recorded safely

Inventory is reduced when an order is accepted.

### 5. Track orders

Customers can view their own orders, while admins can view all orders and update their status to pending, shipped, or delivered.

### 6. Use the price suggestion feature

Admins can ask the ML-powered pricing endpoint for a suggested price based on:

- product category
- current inventory level

The suggestion comes from a saved model and category anchors stored under the PricerMlModel folder.

## API and frontend notes

The backend exposes a Swagger UI at /docs and a ReDoc view at /redoc. These are useful when you want to test the API without building a separate client.

The frontend is intentionally simple and is meant to show how the backend can be consumed in a browser, not to be a complete production UI.

## Testing

The test suite covers the core router behavior.

```bash
pytest -v ./test --tb=long --showlocals
```

## Machine learning piece

The pricing model is stored in the PricerMlModel folder. If you want to retrain it, the project includes commands for preprocessing, training, and testing:

```bash
make ml-preprocess
make ml-train
make ml-test
```

## CI/CD

Every push triggers an automated pipeline via GitHub Actions that:

1. installs dependencies
2. checks code formatting with ruff
3. runs the linter
4. runs the full test suite with pytest

The pipeline runs on all branches so issues are caught before reaching main.
Tests use an in-memory SQLite database so no external services are needed in CI.


## Notes for working with this repo

If you are reading this project as a learning example, the most useful parts to look at first are:

- RetailApp/main.py for the app startup and router setup
- RetailApp/routes for the actual business logic
- RetailApp/schemas.py for the API contract
- PricerMlModel/src for the pricing workflow

That should give you a clear picture of how the app is built and how the pieces fit together.

