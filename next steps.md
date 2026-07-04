Got it — I'll skip the file approach and give it to you directly here in chat, split into parts since it's long. Let's start with Phase 1.

---

# ML Engineer Roadmap — Detailed Tutorial

## Before we begin

This tutorial is meant to be the only reference you need. Everything connects to your existing retail project — four routers, MySQL, JWT auth, Pytest, Loguru. We build on top of that, not from scratch.

---

# PHASE 1 — Step 1: Add a `/predict` endpoint with a machine learning model

### What this step is about and why it matters

Right now your retail app manages users, customers, and orders — that's a backend application. What turns it into an ML application is the ability to also make predictions: take in data about a customer and return a judgment about whether they're likely to stop buying.

This is the single most common task an ML engineer does day to day: take a model that already exists as a trained artifact, and wrap it so other systems can call it over HTTP. Companies don't hire ML engineers to only train models in notebooks — they hire them to make models usable by other software. That's exactly what this step teaches.

### Understanding the architecture before writing code

Walk through what happens when someone calls `/predict/churn`:

A request arrives carrying JSON — info about a customer (orders placed, money spent, days since last purchase). Before touching that data, we must validate it. What if someone sends a negative number? What if a field is missing entirely, or the wrong type? These things *will* happen in production, and if we don't catch them before they reach the model, the model will either crash or silently produce garbage.

Once validated, we run the data through the model. The model was trained earlier, offline, and saved as a file — a frozen snapshot of everything it learned. At request time we just load that file once and reuse it, then call `.predict()` on the new data.

We then take the raw model output (usually just a number) and shape it into something meaningful for the caller: not just "0.77" but "this customer is high risk, 77% probability of churning."

And finally, we handle failure: missing model file, an exception thrown mid-prediction, and so on. All of this — validate, predict, structure, protect against failure — is the full pattern you're about to build.

### Why we load the model once, not per request

Loading a model from disk is comparatively slow — a scikit-learn RandomForest might take a few hundred milliseconds, a larger model could take seconds. If you reload it on every request, every user pays that cost. The fix: load it exactly once, when the app starts, and keep the model sitting in memory for the app's whole lifetime. In Python, a variable defined at the top level of a file (not inside a function) is created once when the module loads and survives as long as the app runs. That's the mechanism we'll use, combined with FastAPI's `lifespan` startup hook.

---

### 1.1 — Train and save a churn model

Create `train_model.py` in your project root. We use synthetic data here so you can run this immediately — swap in your real churn dataset later, the rest of the pipeline doesn't change.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Each row = one customer. Notice the pattern: customers who churned
# tend to have few orders, low spend, and many days since their last order.
data = {
    "total_orders":          [1,5,12,2,8,3,20,1,6,15,2,9,1,4,18],
    "avg_order_value":       [50,200,350,40,180,90,500,30,160,400,60,220,45,110,480],
    "days_since_last_order": [300,10,5,250,20,180,3,400,15,7,280,12,350,160,4],
    "total_spent":           [50,1000,4200,80,1440,270,10000,30,960,6000,120,1980,45,440,8640],
    "churned":               [1,0,0,1,0,1,0,1,0,0,1,0,1,1,0],
}

df = pd.DataFrame(data)
X = df.drop("churned", axis=1)
y = df["churned"]

# test_size=0.2 → 20% of data held out for testing
# random_state=42 → same split every time you run this (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Feature order (IMPORTANT — must match at prediction time): {list(X.columns)}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")
print("Saved to models/churn_model.pkl")
```

Run it:
```bash
pip install scikit-learn joblib pandas
python train_model.py
```

**Critical detail to understand:** the model learned the relationship between features in a specific column order — `[total_orders, avg_order_value, days_since_last_order, total_spent]`. At prediction time, you must feed features back in that exact order. If you swap two columns, the model won't error — it will just silently give wrong answers, because it has no way of knowing the columns got reordered. This is one of the most common real-world bugs in ML serving code, so keep it in mind for step 1.3.

---

### 1.2 — Build the Pydantic schemas

Schemas do two jobs in FastAPI: they declare exactly what shape of data is acceptable, and they validate it automatically before your code ever runs. You already use this pattern in your other routers — here we add a few techniques worth understanding deeply.

`Field(..., ge=0)` — the `...` marks the field required; `ge=0` means "greater than or equal to zero." Pydantic enforces this *before* your function body runs. If it fails, the caller gets a 422 response with a clear explanation — you never write `if x < 0: raise ValueError` manually.

`@field_validator` — for logic that's more than a simple bound. For example, checking that two fields are consistent with each other (you can't have spent money with zero orders).

`Literal["low","medium","high"]` — restricts a field to exactly these three strings, nothing else. Useful for categorical output your downstream code can safely branch on.

Create `schemas/prediction.py`:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class ChurnPredictionInput(BaseModel):
    total_orders: int = Field(..., ge=0, description="Total orders placed")
    avg_order_value: float = Field(..., ge=0.0, description="Average order value in dollars")
    days_since_last_order: int = Field(..., ge=0, description="Days since last order")
    total_spent: float = Field(..., ge=0.0, description="Lifetime spend")

    @field_validator("total_orders")
    @classmethod
    def reasonable_orders(cls, v):
        # Business-rule check, not just a type check — catches obvious data errors
        if v > 100_000:
            raise ValueError("total_orders unreasonably high — likely a data error")
        return v

    @field_validator("total_spent")
    @classmethod
    def spend_consistency(cls, v, info):
        # info.data holds already-validated fields, so we can cross-check them
        if info.data.get("total_orders") == 0 and v > 0:
            raise ValueError("total_spent must be 0 if total_orders is 0")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_orders": 3,
                "avg_order_value": 85.50,
                "days_since_last_order": 120,
                "total_spent": 256.50,
            }
        }
    }


class ChurnPredictionOutput(BaseModel):
    will_churn: bool
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    model_version: str
```

Notice that input and output are two separate classes, even though they're related. They evolve independently — the input shape is about what callers send you, the output shape is a contract you guarantee to honor. Conflating them causes pain later when one needs to change but the other shouldn't.

---

### 1.3 — Build the prediction router

A few mechanics worth understanding before the code:

The `global` keyword. We declare `model = None` at module level. Inside `load_model()`, writing `global model` then `model = joblib.load(...)` tells Python "modify the module-level variable," not "create a new local one that disappears when the function returns." Without `global`, the module-level `model` would stay `None` forever even after calling `load_model()`.

The shape of the feature array. scikit-learn's `.predict()` expects a 2D array: rows are samples, columns are features. Even for one customer, you still need a 2D array with one row. `np.array([[a, b, c, d]])` — the outer brackets make it 2D, the inner brackets are the single row.

`predict_proba()` returns probabilities for each class as `[[prob_class_0, prob_class_1]]`. We want probability of churn (class 1), so we index `[0][1]` — first (only) row, second column.

```python
from fastapi import APIRouter, HTTPException, status
from schemas.prediction import ChurnPredictionInput, ChurnPredictionOutput
import joblib
import numpy as np
import os
from loguru import logger

router = APIRouter(prefix="/predict", tags=["Prediction"])

MODEL_PATH = "models/churn_model.pkl"
MODEL_VERSION = "1.0.0"
model = None  # filled in by load_model() at app startup


def load_model():
    """
    Called once at app startup. Loading is slow (hundreds of ms),
    so we do it once and keep the model in memory for the app's lifetime —
    not on every request.
    """
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        return
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH} ({type(model).__name__})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


def get_risk_level(probability: float) -> str:
    """
    Separated into its own function for two reasons: it's easy to find
    and tune the thresholds in one place, and it's trivially unit-testable
    without needing the model at all.
    """
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    return "high"


@router.post("/churn", response_model=ChurnPredictionOutput, summary="Predict churn")
async def predict_churn(input_data: ChurnPredictionInput):
    # By the time we're here, FastAPI already validated input_data via Pydantic.
    # If validation had failed, this function would never have been called.

    if model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Contact the administrator.",
        )

    # Order MUST match training order — see note in step 1.1
    features = np.array([[
        input_data.total_orders,
        input_data.avg_order_value,
        input_data.days_since_last_order,
        input_data.total_spent,
    ]])

    logger.info(f"Predicting for input: {input_data.model_dump()}")

    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        )

    risk = get_risk_level(float(probability))
    logger.info(f"Result: will_churn={bool(prediction)}, prob={probability:.3f}, risk={risk}")

    return ChurnPredictionOutput(
        will_churn=bool(prediction),
        churn_probability=round(float(probability), 4),
        risk_level=risk,
        model_version=MODEL_VERSION,
    )


@router.get("/health")
async def prediction_health():
    """Health endpoints let monitoring tools detect a broken service automatically."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ready", "model_version": MODEL_VERSION}
```

---

### 1.4 — Wire it into main.py with a proper startup hook

FastAPI's modern way to run startup/shutdown code is the `lifespan` context manager. Code before `yield` runs at startup; code after `yield` runs at shutdown.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers.prediction import router as prediction_router, load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()   # runs once before the app starts accepting requests
    yield
    # any shutdown cleanup would go here

app = FastAPI(title="Retail ML API", lifespan=lifespan)
app.include_router(prediction_router)
# app.include_router(users_router)  -- your existing routers stay as they are
```

If you already have a `lifespan` defined, just add `load_model()` inside it, before the `yield`.

---

### 1.5 — Test it manually, then with Pytest

Run the app, open `/docs`, and POST to `/predict/churn` with:
```json
{"total_orders": 1, "avg_order_value": 40.0, "days_since_last_order": 300, "total_spent": 40.0}
```

Now the automated tests. We load the model once at the top of the test file (not inside each test — that would be slow and pointless repetition). We deliberately test several *categories* of input: valid cases that should clearly classify one way or another, and invalid cases that should be rejected with 422.

```python
import pytest
from fastapi.testclient import TestClient
from main import app
from routers.prediction import load_model, get_risk_level

load_model()
client = TestClient(app)


def test_risk_level_boundaries():
    # Pure logic test — no HTTP, no model needed
    assert get_risk_level(0.1) == "low"
    assert get_risk_level(0.5) == "medium"
    assert get_risk_level(0.9) == "high"


def test_predict_response_shape():
    response = client.post("/predict/churn", json={
        "total_orders": 5, "avg_order_value": 100.0,
        "days_since_last_order": 30, "total_spent": 500.0,
    })
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"will_churn", "churn_probability", "risk_level", "model_version"}
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_high_risk_customer():
    response = client.post("/predict/churn", json={
        "total_orders": 1, "avg_order_value": 40.0,
        "days_since_last_order": 350, "total_spent": 40.0,
    })
    assert response.json()["risk_level"] == "high"


def test_predict_rejects_negative_orders():
    response = client.post("/predict/churn", json={
        "total_orders": -1, "avg_order_value": 100.0,
        "days_since_last_order": 30, "total_spent": 100.0,
    })
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    response = client.post("/predict/churn", json={
        "total_orders": 5, "avg_order_value": 100.0,
        "total_spent": 500.0,  # days_since_last_order missing
    })
    assert response.status_code == 422


def test_health_ready():
    response = client.get("/predict/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
```

Run with: `pytest tests/test_prediction.py -v`

---

Here's the rest, continuing straight through.

---

# PHASE 1 — Step 2: Switch from MySQL to PostgreSQL and add Alembic migrations

### What this step is about and why it matters

This step has two related but distinct parts. First, moving to PostgreSQL aligns you with what the overwhelming majority of professional backend systems use. Not because MySQL is bad — it isn't — but because if you work at most companies, their production database will be PostgreSQL, and knowing exactly where SQLAlchemy's abstraction holds and where it doesn't is genuinely useful knowledge, not just resume padding.

Second, adding Alembic solves a real problem you probably have right now: when you need to add a column, you likely drop and recreate the table, or hand-write SQL. That's fine when you're the only developer working against your own local database. It is not fine on a team, and it is absolutely not fine in production where real data lives. Alembic tracks every schema change as a versioned, reversible Python file.

### Why PostgreSQL specifically

PostgreSQL has first-class JSON/JSONB support with indexing inside JSON documents — relevant if you ever store model outputs or feature vectors directly in a column. It has a richer native type system (arrays, ranges, custom types) that MySQL lacks. Its concurrency model (MVCC) handles many simultaneous reads and writes more gracefully, which matters once your `/predict` endpoint is getting real traffic. And every major managed cloud database service treats PostgreSQL as a first-class citizen.

For your code, none of this requires rewriting anything. SQLAlchemy's Core layer generates the SQL dialect appropriate to whichever database you're connected to — `String(100)` becomes `VARCHAR(100)` in both. The only thing that changes is your connection string. The abstraction breaks down only for very database-specific syntax (MySQL's `ON DUPLICATE KEY UPDATE` vs PostgreSQL's `ON CONFLICT DO UPDATE`), which you're not using.

---

### 2.1 — Install the driver

```bash
pip install psycopg2-binary alembic
```

`psycopg2-binary` is PostgreSQL's Python driver, the equivalent of `pymysql`. The `-binary` build includes precompiled C extensions — faster to install, slightly larger. Fine for now; in a size-sensitive production image you'd eventually switch to compiling `psycopg2` from source, but don't worry about that yet.

---

### 2.2 — Update your database configuration

Replace your `database.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os

# Reading from an environment variable (with a sane local default) means
# the exact same code works unmodified in three different contexts:
# locally on your laptop, inside Docker Compose, and on an EC2 server —
# only the environment variable's value changes between them.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:password@localhost:5432/retail_db"
)

# pool_pre_ping=True makes SQLAlchemy test a connection before handing it
# out from the pool. Database connections can silently go stale — a
# restart on the DB side, a network blip — and without this you'd see
# random "connection was closed" errors that are maddening to debug.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """All models inherit from this. Alembic reads Base.metadata to know
    what tables should exist."""
    pass


def get_db():
    """
    FastAPI dependency providing one session per request.
    The try/finally guarantees db.close() runs even if the route handler
    raises an exception — otherwise you'd leak connections out of the pool
    until it's exhausted.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### 2.3 — What Alembic actually solves, in concrete terms

Imagine two developers on the same project. Developer A adds a column locally. Developer B pulls the new code the next day, runs the app, and it crashes — their database doesn't have the new column, because schema changes aren't part of git history by default; only the model code is. Now imagine the same scenario in production: someone has to manually write and run ALTER TABLE statements against a database holding real customer data, hoping nothing goes wrong and there's no way to cleanly undo it if something does.

Alembic turns every schema change into a migration file with an `upgrade()` function (apply it) and a `downgrade()` function (undo it). Migrations are ordered and tracked in a special table Alembic creates (`alembic_version`), so it always knows which migrations have already been applied to a given database. When Developer B pulls the new code, they run one command — `alembic upgrade head` — and their local database catches up automatically. The exact same command is what you run in production during deployment.

The `--autogenerate` flag is what makes this painless day to day: Alembic compares your SQLAlchemy models against the current state of the actual database and writes the migration code for you. You still read and approve it before applying — autogenerate is a draft, not a blind trust mechanism — but you almost never hand-write migrations for ordinary column changes.

---

### 2.4 — Set up Alembic

```bash
alembic init alembic
```

This creates `alembic/env.py` (the migration runner's configuration), `alembic/versions/` (where generated migration files live), and `alembic.ini` (top-level config).

Edit `alembic/env.py`:

```python
# Add near the top, before the existing imports
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Base
import models.user        # import every model module — Alembic needs all
import models.customer    # of them imported so Base.metadata knows every
import models.order       # table that should exist. Forgetting one means
                           # autogenerate silently won't see that table.

# Find this line and change it:
target_metadata = Base.metadata
```

And inside `run_migrations_online()` in the same file:

```python
def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:password@localhost:5432/retail_db"
    )
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
```

This makes Alembic read the same `DATABASE_URL` environment variable your app uses, so migrations always target the correct database no matter where you run them.

---

### 2.5 — Generate and apply your first migration

With PostgreSQL running (locally installed, or via the Compose file you'll write in Step 3):

```bash
alembic revision --autogenerate -m "initial schema"
```

This produces a file in `alembic/versions/` like `a3f1d2b4c5e6_initial_schema.py`. Open it and actually read it:

```python
def upgrade() -> None:
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=200), nullable=False),
        sa.Column('hashed_password', sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
    )
    # ... more tables, mirroring your models

def downgrade() -> None:
    op.drop_table('users')
    # ...
```

This is Alembic translating your Python model classes into the SQL needed to create matching tables. Apply it:

```bash
alembic upgrade head
```

`head` means "apply every migration up to the latest one." Your tables now exist in PostgreSQL.

---

### 2.6 — The workflow you'll repeat for every future schema change

Memorize this four-step loop, because you'll use it constantly:

1. Edit the SQLAlchemy model in Python.
2. `alembic revision --autogenerate -m "describe the change"`
3. Open the generated file and read it.
4. `alembic upgrade head`

Concrete example — adding `loyalty_points` to your Customer model:

```python
class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    loyalty_points = Column(Integer, default=0)  # new line
```

```bash
alembic revision --autogenerate -m "add loyalty_points to customers"
```

Generated migration:

```python
def upgrade() -> None:
    op.add_column('customers', sa.Column('loyalty_points', sa.Integer(), nullable=True))

def downgrade() -> None:
    op.drop_column('customers', 'loyalty_points')
```

```bash
alembic upgrade head
```

Done. No dropped tables, no hand-written SQL, no risk to existing rows. And if you decide the change was wrong, `alembic downgrade -1` undoes exactly this migration and nothing else.

---

### 2.7 — Command reference, for when you're working day to day

```bash
alembic upgrade head              # apply all pending migrations
alembic downgrade -1              # undo the most recent migration
alembic downgrade base            # undo everything, back to empty
alembic current                   # show which migration is currently applied
alembic history                   # show the full migration history
alembic revision --autogenerate -m "message"   # generate from model diffs
alembic revision -m "message"     # create an empty migration (manual SQL)
```

---

# PHASE 1 — Step 3: Docker Compose

### What this step is about and why it matters

Docker Compose lets you define your entire multi-service application — FastAPI, PostgreSQL, Redis — in one YAML file and bring it all up with one command, instead of manually starting each piece in separate terminals and wiring up connection details by hand. This matters for three reasons: it makes your dev environment reproducible for anyone who clones your repo, it eliminates "works on my machine" entirely, and it's structurally very close to how production deployments work, so the jump to AWS later is small.

### Images vs containers — the mental model you need first

A Docker **image** is a read-only template: an OS layer, your code, your dependencies, and instructions for running it. Images are immutable once built. A **container** is a running instance of an image — like a process with its own isolated filesystem and network. You can run many containers from one image. When a container stops, anything it wrote to its own filesystem disappears unless you used a volume (more on that below).

A **Dockerfile** is the recipe used to build an image. Each line is a build step, and Docker caches the result of each step — if a step's inputs haven't changed since the last build, Docker reuses the cached result instead of redoing the work. This is why the *order* of instructions in a Dockerfile affects build speed, which is why we install dependencies before copying application code (explained below).

### How Docker Compose networking actually works

When Compose starts multiple services, it creates a private network connecting them, and on that network, **each service can reach the others using the service's name as a hostname**. If your PostgreSQL service is named `postgres` in the Compose file, your FastAPI container connects to it via the hostname `postgres` — never `localhost`. Inside the FastAPI container, `localhost` means the FastAPI container itself, not the PostgreSQL container. This is the single most common point of confusion for people new to Compose, so it's worth internalizing now: `@postgres:5432` in a connection string means "the container literally named `postgres`, port 5432," and Docker resolves that name to the right container's IP automatically.

### Why volumes matter

By default, anything inside a container is temporary — if PostgreSQL's container restarts, its data is gone. A named **volume** is storage managed by Docker that lives outside any individual container's lifecycle. We mount a volume at the directory where PostgreSQL stores its data files, so that data survives container restarts and rebuilds.

### Why healthchecks and depends_on matter together

Compose starts services in parallel by default, which creates a race: your API might try to connect to PostgreSQL before PostgreSQL is actually ready to accept connections, and crash with "connection refused." A healthcheck defines a command Compose runs periodically to verify a service is truly ready — `pg_isready` for PostgreSQL actually attempts a connection and reports success or failure. `depends_on` with `condition: service_healthy` tells Compose "don't start this service until that one's healthcheck passes," not just "until it starts."

---

### 3.1 — Write the Dockerfile

```dockerfile
# "slim" — official Python build with only what's needed to run Python,
# no compilers or docs. Smaller image, faster pulls.
FROM python:3.11-slim

# Skip writing .pyc bytecode files (no benefit inside a container)
ENV PYTHONDONTWRITEBYTECODE=1
# Force stdout/stderr to flush immediately — without this, logs can
# appear delayed or out of order when you run `docker compose logs`
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copying requirements.txt BEFORE the rest of the code is the single
# most important Docker optimization here, and it's about layer caching:
# Docker caches each instruction's result. If requirements.txt hasn't
# changed since the last build, Docker reuses the cached "pip install"
# layer instead of re-running it — even though your .py files changed.
# If we copied everything first, ANY code change would force a full
# package reinstall every time, which is slow.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# --host 0.0.0.0 is REQUIRED inside Docker. Without it, uvicorn only
# listens on 127.0.0.1 *inside the container*, which makes it completely
# unreachable from outside — even though the container is "running fine."
# 0.0.0.0 means "listen on every network interface."
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 3.2 — Write docker-compose.yml

```yaml
version: "3.9"

services:

  postgres:
    image: postgres:15-alpine
    container_name: retail_postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: retail_db
    volumes:
      # Named volume → data survives `docker compose down` and rebuilds.
      # Without this, every restart wipes your database.
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"   # HOST:CONTAINER — lets you connect from pgAdmin/DBeaver locally
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: retail_redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: retail_api
    ports:
      - "8000:8000"
    environment:
      # Hostname is "postgres" — the SERVICE NAME, not "localhost".
      # This is the networking point explained above.
      DATABASE_URL: postgresql+psycopg2://postgres:password@postgres:5432/retail_db
      REDIS_URL: redis://redis:6379
      SECRET_KEY: change-this-in-production
      ENVIRONMENT: development
    depends_on:
      postgres:
        condition: service_healthy   # wait for the healthcheck, not just container start
      redis:
        condition: service_healthy
    volumes:
      # Bind mount: your local files ARE the container's files. Edit on
      # your laptop, see it instantly inside the container. Combined with
      # --reload below, the server restarts automatically on save.
      # Remove this mount in production — there, code should be baked
      # into the image, not mounted from outside.
      - .:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  migrate:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: retail_migrate
    environment:
      DATABASE_URL: postgresql+psycopg2://postgres:password@postgres:5432/retail_db
    depends_on:
      postgres:
        condition: service_healthy
    command: alembic upgrade head   # overrides the Dockerfile's default CMD
    restart: "no"                   # runs once, exits, and we don't want it restarted

volumes:
  postgres_data:
```

---

### 3.3 — Essential commands

```bash
docker compose up              # start everything, foreground, see all logs
docker compose up -d           # same, but detached (get your terminal back)
docker compose down            # stop + remove containers, KEEP volumes (data safe)
docker compose down -v         # stop + remove containers AND delete volumes (data gone)
docker compose up --build      # force rebuild — use after changing requirements.txt/Dockerfile
docker compose ps              # see what's running
docker compose logs -f api     # follow one service's logs in real time
docker compose exec api bash   # open a shell inside the running api container
docker compose exec postgres psql -U postgres retail_db   # connect to the DB directly
```

---

### 3.4 — Add Redis caching to the prediction endpoint

The idea: before running the model, check whether we've already computed this exact prediction recently. If yes, return the cached result instantly. If no, run the model, store the result with an expiry, return it. This matters because if a dashboard or another service calls `/predict/churn` repeatedly for the same customer (e.g. on every page refresh), there's no reason to burn CPU re-running the model each time.

```bash
pip install redis
```

```python
import redis
import json
import os
import hashlib

redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,
)
CACHE_TTL_SECONDS = 300  # 5 minutes, then Redis auto-expires the entry


def make_cache_key(input_data: ChurnPredictionInput) -> str:
    # sort_keys=True ensures the same input always produces the same JSON
    # string regardless of field insertion order, so the hash is stable.
    data_str = json.dumps(input_data.model_dump(), sort_keys=True)
    return f"churn:{hashlib.md5(data_str.encode()).hexdigest()}"


@router.post("/churn", response_model=ChurnPredictionOutput)
async def predict_churn(input_data: ChurnPredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    cache_key = make_cache_key(input_data)

    try:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            return ChurnPredictionOutput(**json.loads(cached))
    except redis.RedisError as e:
        # A cache outage must never break predictions — log and fall through.
        logger.warning(f"Redis unavailable, skipping cache: {e}")

    features = np.array([[
        input_data.total_orders, input_data.avg_order_value,
        input_data.days_since_last_order, input_data.total_spent,
    ]])

    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    result = ChurnPredictionOutput(
        will_churn=bool(prediction),
        churn_probability=round(float(probability), 4),
        risk_level=get_risk_level(float(probability)),
        model_version=MODEL_VERSION,
    )

    try:
        redis_client.setex(cache_key, CACHE_TTL_SECONDS, result.model_dump_json())
    except redis.RedisError as e:
        logger.warning(f"Failed to cache result: {e}")

    return result
```

---

# PHASE 1 — Step 4: GitHub Actions — add Continuous Deployment

### What this step is about and why it matters

You already have CI: tests run on every push. CD extends that: when tests pass *and* the branch is `main`, automatically build a Docker image and push it to Docker Hub. Right now, shipping a new version means manually SSHing into a server, pulling code, restarting things by hand — slow and error-prone. With CD, every merged change is automatically packaged and ready to deploy with a single pull command on the server. It also signals something specific to hiring managers: you understand that code isn't "done" when it runs on your laptop — it's done when it's tested, packaged, and deployable anywhere.

### The GitHub Actions concepts you need

A **workflow** is a YAML file under `.github/workflows/`. GitHub auto-detects every file there. A workflow is triggered by an **event** — `on: push`, `on: pull_request`, etc. A workflow contains **jobs**, which can run in parallel or be chained with `needs:`. Each job runs on a fresh virtual machine. A job contains **steps**, which run sequentially — either a shell command (`run:`) or a reusable **action** (`uses:`), which is prebuilt code published by someone else (GitHub itself, or Docker, in our case) for a common task like "checkout the repo" or "log in to Docker Hub."

GitHub Actions also supports **service containers** — Docker containers that run alongside your job, providing things like a real database for tests, without needing it pre-installed on the runner.

---

### 4.1 — Set up Docker Hub credentials as GitHub Secrets

Your workflow needs to authenticate with Docker Hub, but you can't put a password directly in a workflow file — that file is visible to anyone who can see your repo. GitHub Secrets are encrypted values you set once in repo settings and reference as `${{ secrets.NAME }}`; they never appear in logs.

Create a Docker Hub access token at hub.docker.com → Account Settings → Security → New Access Token (copy it immediately, it's shown only once). Then in your GitHub repo: Settings → Secrets and variables → Actions → New repository secret. Add:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

### 4.2 — Create the CD workflow

`.github/workflows/cd.yml`:

```yaml
name: CD — Build and push Docker image

on:
  push:
    branches:
      - main   # only main — feature branches push freely without triggering a build

jobs:

  test:
    name: Run tests before building
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: password
          POSTGRES_DB: retail_test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - run: pip install -r requirements.txt

      - name: Train the model (tests need it present)
        run: python train_model.py

      - name: Apply migrations to the test database
        env:
          DATABASE_URL: postgresql+psycopg2://postgres:password@localhost:5432/retail_test_db
        run: alembic upgrade head

      - name: Run tests
        env:
          DATABASE_URL: postgresql+psycopg2://postgres:password@localhost:5432/retail_test_db
          REDIS_URL: redis://localhost:6379
          SECRET_KEY: test-secret
          ENVIRONMENT: test
        run: pytest tests/ -v --tb=short

  build-and-push:
    name: Build and push Docker image
    runs-on: ubuntu-latest
    needs: test   # only runs if the test job above succeeds — never push an untested image

    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Generate tags
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ secrets.DOCKERHUB_USERNAME }}/retail-api
          tags: |
            type=raw,value=latest
            type=sha,prefix=sha-

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

Two tags get pushed: `latest` (always the newest build from main) and `sha-<commit hash>` — the second one lets you say precisely "production is running commit a3f2c1d" and trace exactly what code that is, which matters when debugging a production issue later.

Push to main, check the Actions tab on GitHub to watch it run, then check hub.docker.com — you'll see your image listed.

---

# PHASE 2 — Step 5: Deploy to AWS EC2

### What this step is about and why it matters

Everything so far runs on your laptop. This step puts it on a real, internet-reachable server. We use three AWS services: **EC2** (a rented virtual machine — where your containers actually run), **S3** (object storage — where we'll keep the trained model file so the app downloads it at startup rather than baking it into the image), and an **IAM role** (the correct, credential-free way to grant your server permission to read from S3).

### Why store the model in S3 instead of just putting it in the Docker image

You *can* just copy the model file into the image — simpler to start with. But ML models can be large, and baking them into the image makes every push/pull slower, and worse: retraining the model means rebuilding the entire image just to update one file. With S3, you upload a new model file and just restart the containers — no image rebuild. For your first deployment, either approach is fine; this tutorial shows the S3 approach because it's the pattern you'll actually use once models get bigger.

---

### 5.1 — Install and configure the AWS CLI

```bash
pip install awscli
```

Create an access key: AWS Console → IAM → Users → your user → Security credentials → Create access key → choose "Command Line Interface."

```bash
aws configure
```
It asks for Access Key ID, Secret Access Key, default region (`eu-west-1` is a reasonable choice from Egypt), and output format (`json`).

Verify:
```bash
aws sts get-caller-identity
```

---

### 5.2 — Store the model in S3

Bucket names are globally unique across *all* AWS accounts, so prefix yours with something personal.

```bash
aws s3 mb s3://your-name-retail-models --region eu-west-1
aws s3 cp models/churn_model.pkl s3://your-name-retail-models/churn_model.pkl
aws s3 ls s3://your-name-retail-models/
```

Update `load_model()` to be environment-aware — use the local file in development, download from S3 in production:

```python
import boto3

def load_model():
    global model
    os.makedirs("models", exist_ok=True)
    environment = os.getenv("ENVIRONMENT", "development")

    if environment == "development" and os.path.exists(MODEL_PATH):
        logger.info(f"Dev mode: loading local model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        return

    bucket = os.getenv("S3_BUCKET")
    s3_key = os.getenv("MODEL_S3_KEY", "churn_model.pkl")
    if not bucket:
        logger.error("S3_BUCKET not set and no local model file found.")
        return

    logger.info(f"Downloading model from s3://{bucket}/{s3_key}")
    try:
        boto3.client("s3").download_file(bucket, s3_key, MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("Model downloaded and loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download model from S3: {e}")
```

```bash
pip install boto3
```

---

### 5.3 — Launch the EC2 instance

In the AWS Console, EC2 → Launch Instance:
- AMI: Ubuntu Server 22.04 LTS
- Instance type: `t2.micro` (free tier — 1 CPU, 1GB RAM, enough for this)
- Key pair: create new, name it `retail-key`, download the `.pem` file, keep it safe — you cannot redownload it
- Security group inbound rules: SSH (port 22) from "My IP", Custom TCP port 8000 from "Anywhere"

Launch, wait about a minute, note the instance's public IPv4 address.

---

### 5.4 — Connect and install Docker

```bash
chmod 400 ~/.ssh/retail-key.pem
ssh -i ~/.ssh/retail-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

Inside the server:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu   # lets you run docker without sudo
newgrp docker
docker --version
docker compose version
```

---

### 5.5 — Grant S3 access via an IAM role (not credentials on the server)

The correct AWS way to let EC2 read from S3 is an IAM role attached to the instance — never put access keys directly on a server.

AWS Console → IAM → Roles → Create role → Trusted entity: AWS service → EC2 → attach policy `AmazonS3ReadOnlyAccess` → name it `retail-api-s3-read`.

Then attach it: EC2 → select your instance → Actions → Security → Modify IAM role → select `retail-api-s3-read`.

Once attached, boto3 on the instance automatically picks up these permissions — no keys, no `aws configure` needed on the server itself.

---

### 5.6 — Deploy

On the EC2 server:

```bash
mkdir -p /home/ubuntu/retail_app && cd /home/ubuntu/retail_app
```

Create `docker-compose.prod.yml`:

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: choose-a-strong-password
      POSTGRES_DB: retail_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine

  api:
    image: YOURDOCKERHUBUSERNAME/retail-api:latest
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+psycopg2://postgres:choose-a-strong-password@postgres:5432/retail_db
      REDIS_URL: redis://redis:6379
      SECRET_KEY: choose-a-long-random-secret
      ENVIRONMENT: production
      S3_BUCKET: your-name-retail-models
      MODEL_S3_KEY: churn_model.pkl
    depends_on:
      postgres:
        condition: service_healthy

  migrate:
    image: YOURDOCKERHUBUSERNAME/retail-api:latest
    environment:
      DATABASE_URL: postgresql+psycopg2://postgres:choose-a-strong-password@postgres:5432/retail_db
    depends_on:
      postgres:
        condition: service_healthy
    command: alembic upgrade head
    restart: "no"

volumes:
  postgres_data:
```

```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs api
```

Your app is now live at `http://YOUR_EC2_PUBLIC_IP:8000/docs`.

---

### 5.7 — Updating the deployment after future pushes

Every push to main triggers GitHub Actions to build and push a new `latest` image. To deploy that update:

```bash
# on the EC2 server
cd /home/ubuntu/retail_app
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

This pulls the new image and restarts only what changed. Your PostgreSQL data is untouched — it lives in the named volume, completely separate from the application containers.

---

# PHASE 3 — Step 6: MLflow tracking, properly integrated

### What this step is about and why it matters

You already know MLflow from your coursework, but using it for real means understanding what it actually tracks. An **experiment** is a named group of related training **runs**. A run is one execution of your training code, and MLflow records the parameters you used, the metrics that resulted, and any artifacts (like the model file) produced — all linked together, all comparable side by side later, and all reproducible. This is the difference between "I trained a model once and saved the file" and "I have a record of every experiment I ran and why I chose this one."

```python
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("mlruns")          # local folder; fine for a portfolio project
mlflow.set_experiment("churn-prediction")  # creates the experiment if it doesn't exist

params = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42,
}

with mlflow.start_run(run_name=f"rf_n{params['n_estimators']}"):
    mlflow.log_params(params)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("features", list(X.columns))

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "churn_model")

    joblib.dump(model, "models/churn_model.pkl")
    mlflow.log_artifact("models/churn_model.pkl", artifact_path="joblib_model")

    print(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
```

```bash
pip install mlflow
mlflow ui
# open http://localhost:5000
```

Now every time you retrain — change a hyperparameter, add a feature — you get a new run you can compare against past ones, instead of overwriting history and losing track of what worked.

---

# PHASE 3 — Step 7: Write a README that actually tells the story

### Why this matters more than it seems

A hiring manager opening your repository decides in about thirty seconds whether to look closer. Most developer READMEs are either empty or just installation commands. A good one for a portfolio project answers, without requiring the reader to open any code: what does this do, what's it built with and why, how is it architected, and how do I run it. It demonstrates you can communicate technically — a skill that matters as much as the code itself once you're past the junior level.

```markdown
# Retail ML API

A retail management API with integrated customer churn prediction.
FastAPI, PostgreSQL, Redis, scikit-learn, Docker.

## Architecture

User Request
    │
    ▼
FastAPI Application
    ├── /api/users       → PostgreSQL
    ├── /api/auth        → JWT (bcrypt + python-jose)
    ├── /api/customers   → PostgreSQL
    ├── /api/orders      → PostgreSQL
    └── /predict/churn   → Redis cache → scikit-learn model → S3 (model storage)

CI/CD (GitHub Actions)
    ├── every push:    pytest
    └── merge to main: Docker build → Docker Hub → EC2

MLflow Tracking
    └── experiment runs, parameters, metrics, model artifacts

## Running locally

git clone https://github.com/yourusername/retail-api
cd retail-api
python train_model.py
docker compose up --build

Open http://localhost:8000/docs

## Tests

pytest tests/ -v

## Deployment

Deployed on AWS EC2 via Docker Compose. The model is stored in S3 and
downloaded on startup. Every merge to main triggers an automatic
Docker build and push via GitHub Actions.
```

---

# PHASE 4 — Step 8: LeetCode, the four patterns that cover most MLE interviews

### What you actually need to understand about these interviews

Most ML engineer interviews include one or two coding problems, and the bar is not "find a clever trick nobody's seen" — it's "write clean, correct Python under mild time pressure, using a recognizable pattern." A small number of patterns cover the large majority of what you'll be asked. Learn the pattern deeply, and the specific problem becomes mostly mechanical.

---

### Pattern 1: Hashmaps

The core insight: any time your algorithm is "for each element, search through the rest of the list for something," you're doing O(n) work inside an O(n) loop — O(n²) total. A hashmap turns "search" into "look up," which is O(1), bringing the whole thing down to O(n).

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    nums = [2, 7, 11, 15], target = 9 → [0, 1] since nums[0]+nums[1]=9

    Naive: nested loop checking every pair — O(n²).
    Hashmap: for each number, ask "what number would complete this pair?"
    (complement = target - num). If we've already seen that complement,
    we're done. We record each number's index as we scan, so the lookup
    is O(1) and the whole algorithm is O(n).
    """
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# i=0, num=2: complement=7, not seen yet. seen={2:0}
# i=1, num=7: complement=2, IS in seen → return [0, 1]
```

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Two words are anagrams iff their sorted characters match exactly.
    "eat" sorted → "aet". "tea" sorted → "aet". Same key, grouped together.
    We use a tuple (not a list) as the key because dict keys must be
    hashable, and lists aren't.
    """
    groups = defaultdict(list)
    for word in strs:
        key = tuple(sorted(word))
        groups[key].append(word)
    return list(groups.values())

print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
# [['eat','tea','ate'], ['tan','nat'], ['bat']]
```

---

### Pattern 2: Sliding Window

Applies whenever you need something over a *contiguous* subarray. The insight: when the window slides one position, only two elements change — one leaves on the left, one enters on the right. There's no need to recompute the whole window from scratch each time.

```python
def max_sum_subarray(nums: list[int], k: int) -> int:
    """
    Max sum of any contiguous subarray of length exactly k.
    nums=[2,1,5,1,3,2], k=3 → 9 (the subarray [5,1,3])

    Naive: re-sum k elements for every starting position — O(n*k).
    Sliding window: compute the first window once, then each step just
    adds the incoming element and subtracts the outgoing one — O(1) per
    step, O(n) total.
    """
    if len(nums) < k:
        return 0
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i]
        window_sum -= nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

```python
def length_of_longest_substring(s: str) -> int:
    """
    Longest substring with no repeated characters.
    "abcabcbb" → 3 ("abc")

    Variable-size window: grow by adding characters on the right; when a
    duplicate appears, shrink from the left until the duplicate is gone.
    A set tracks what's currently "in the window."
    """
    char_set = set()
    left = 0
    max_length = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    return max_length
```

---

### Pattern 3: Two Pointers

Applies on a sorted array when you're looking for pairs/triplets meeting some condition. Instead of checking every pair (O(n²)), two pointers starting at opposite ends and converging finds the answer in O(n): if the current sum is too small, you need a bigger number, and the only way to get one is moving the left pointer forward; if too big, move the right pointer backward.

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    """
    All unique triplets summing to zero.
    [-1,0,1,2,-1,-4] → [[-1,-1,2],[-1,0,1]]

    Sort first, then for each element as the "anchor," use two pointers
    on the remainder of the array to find pairs summing to -anchor.
    Skipping duplicate values at each level avoids duplicate triplets.
    """
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        target = -nums[i]
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1
    return result
```

---

### Pattern 4: Trees and Recursion

Nearly every tree problem follows the same template: handle the base case (usually `node is None`), recursively solve for the left subtree, recursively solve for the right subtree, then combine the two results. Once you can identify "what's the base case" and "how do I combine left and right," most tree problems fall into place.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root: TreeNode) -> int:
    """Empty tree → depth 0. Otherwise 1 + the deeper of the two children."""
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def is_valid_bst(root: TreeNode) -> bool:
    """
    A BST requires every left-subtree value < node < every right-subtree
    value. We pass down a (min, max) bound that tightens as we go deeper:
    going left lowers the max allowed; going right raises the min allowed.
    """
    def validate(node, min_val, max_val):
        if node is None:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    return validate(root, float('-inf'), float('inf'))
```

```python
from collections import deque

def level_order(root: TreeNode) -> list[list[int]]:
    """
    Values grouped by level. Uses BFS with a queue rather than recursion —
    we track how many nodes are currently in the queue (= current level's
    size) so we know exactly when one level ends and the next begins.
    """
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(current_level)
    return result
```

---

### Basic Dynamic Programming

DP applies when a problem's answer for size *n* can be built from answers to smaller sizes, and those smaller answers get reused many times — so we compute each one once, store it, and look it up instead of recomputing.

```python
def climb_stairs(n: int) -> int:
    """
    1 or 2 steps at a time — how many distinct ways to reach step n?
    To reach step n, you arrived either from step n-1 (one step) or
    step n-2 (two steps): ways(n) = ways(n-1) + ways(n-2) — Fibonacci.
    """
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def house_robber(nums: list[int]) -> int:
    """
    Maximize stolen money; adjacent houses can't both be robbed.
    At house i: either skip it (take dp[i-1]) or rob it
    (take nums[i] + dp[i-2], since house i-1 is now off-limits).
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    dp = [0] * len(nums)
    dp[0], dp[1] = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
    return dp[-1]
```

---

### Your weekly practice plan

**Weeks 1–2 — Hashmaps:** LeetCode 1 (Two Sum), 217 (Contains Duplicate), 242 (Valid Anagram), 49 (Group Anagrams), 347 (Top K Frequent Elements)

**Weeks 3–4 — Sliding Window:** LeetCode 121 (Best Time to Buy and Sell Stock), 3 (Longest Substring Without Repeating Characters), 567 (Permutation in String), 76 (Minimum Window Substring — harder)

**Weeks 5–6 — Trees:** LeetCode 104 (Maximum Depth), 226 (Invert Binary Tree), 100 (Same Tree), 572 (Subtree of Another Tree), 102 (Level Order Traversal), 98 (Validate BST)

**Weeks 7–8 — Basic DP:** LeetCode 70 (Climbing Stairs), 198 (House Robber), 322 (Coin Change), 300 (Longest Increasing Subsequence)

Spend 20–30 minutes per problem. Stuck past 25 minutes? Read the solution, understand it fully, close it, then write it yourself from memory. That last step — rewriting from memory — is what actually makes the pattern stick.

---

# Quick reference

**Environment variables**, never hardcoded:
```python
import os
from dotenv import load_dotenv
load_dotenv()  # reads .env locally; no-op in Docker/EC2 where vars are set directly

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
```

`.gitignore` must include: `.env`, `__pycache__/`, `mlruns/`, `models/*.pkl`

**Pre-push checklist:** tests pass → no hardcoded credentials → `.env` not staged → `requirements.txt` current → `docker compose up --build` works clean → `/predict/health` returns ready → `/docs` shows all endpoints.

**The 60-second interview answer:**

"My retail ML API has four domain routers — users, authentication, customers, orders — backed by PostgreSQL with Alembic managing schema migrations. On top of that I added a prediction service: a `/predict/churn` endpoint serving a scikit-learn model trained on customer behavior data, with Redis caching so repeated requests are instant. The model is stored in S3 and downloaded on startup. The whole stack runs in Docker Compose locally. GitHub Actions handles CI with a real PostgreSQL service container, and CD builds and pushes a Docker image to Docker Hub on every merge to main, deployed on AWS EC2. Training runs are tracked with MLflow."

That's the whole roadmap. Let me know which part you want to start implementing first, or if anything needs more depth before you move on.
You're right to flag this — the roadmap so far gets you to "can deploy one model behind one endpoint," which is real and valuable, but it's not yet MLOps in the fuller sense: pipelines that retrain automatically, monitoring that catches drift before it causes damage, and tracking that scales beyond one person's laptop. Let me actually search for what's current here rather than rely on stale knowledge, since tooling in this space moves fast.Good — this confirms what you suspected: what you've built so far (model + endpoint + Docker + CD) is the *foundation*, but real MLOps adds three layers on top of it: **pipelines** (automated retraining triggered by schedule or data change), **monitoring** (catching when the model quietly gets worse in production), and **orchestration** (coordinating multi-step workflows reliably). Let me lay out exactly how to learn each, building directly on your retail project.

---

## Why this feels abstract right now

You're not missing intelligence here — these terms genuinely *are* vague until you build one small example of each. MLOps unites data science with modern software operations through automation, reproducibility, and governance across the model lifecycle, but that sentence means nothing until you've actually watched a pipeline retrain a model on a schedule, or watched a dashboard flag that your model's accuracy quietly dropped. So we'll build one of each, small and concrete, using your churn model.

---

## Layer 1 — Pipelines (the part where retraining stops being manual)

**What it actually is:** Right now, you run `python train_model.py` by hand whenever you want a new model. A pipeline replaces "by hand" with "automatically, on a trigger." An AI deployment pipeline is a structured, automated process that takes a model from training to validation, containerization, testing, deployment, monitoring, and continuous retraining — without breaking under real-world traffic.

**Tool to learn:** Apache Airflow, or its simpler modern cousin, Prefect. Orchestration coordinates complex ML workflows — task scheduling, dependencies, retries, and distributed execution — so pipelines run reliably, with Airflow and Prefect being popular choices. Start with Prefect — it's far less painful to set up locally than Airflow, and the concepts transfer directly.

**Concretely, what you'll build on your retail project:** a pipeline with steps — pull fresh order data from PostgreSQL → retrain the churn model → evaluate it against the current production model → if better, push the new model to S3 → log everything to MLflow. You schedule it (say, weekly) instead of running `train_model.py` by hand.

**How to learn it, step by step:**
1. Install Prefect (`pip install prefect`), do their official "first flow" quickstart — it's about 30 minutes and gets a scheduled Python function running.
2. Convert your `train_model.py` into a Prefect flow with `@flow` and `@task` decorators on each step (load data, train, evaluate, save).
3. Add a comparison step: load the *current* production model's saved metrics, compare against the new model's metrics, only promote if better.
4. Schedule it to run weekly using Prefect's deployment scheduling.

This alone — one working scheduled pipeline — is the single most convincing thing you can show in an interview, more convincing than knowing the word "Airflow."

---

## Layer 2 — Monitoring (catching when the model quietly degrades)

**What it actually is:** Your model was trained on data from one point in time. Real customers change behavior over time — that's called **drift**. MLOps architectures can incorporate data drift detection, concept drift analysis, and model performance tracking, ensuring models remain accurate and reliable as data evolves. Without monitoring, a model can silently get worse for months before anyone notices, because it never crashes — it just quietly gives worse answers.

**Tools to learn, in order of how directly they apply to you:**

*Evidently AI* — purpose-built for ML drift detection, free, and works directly with pandas DataFrames. This is your starting point.

*Prometheus + Grafana* — for infrastructure-level monitoring (is the API slow? Is it erroring?). Tools like Prometheus for metrics collection and Grafana for real-time visualization track model and infrastructure health, with dashboards, alerts, and end-to-end visibility enabling proactive issue detection.

**Concretely, what you'll build:** Take your training data (the "reference" distribution) and a batch of newer, simulated "incoming" customer data (the "current" distribution). Run Evidently's drift report comparing them. It produces an actual HTML report showing exactly which features have drifted and by how much.

**How to learn it, step by step:**
1. `pip install evidently`
2. Generate two CSVs from your synthetic data — one as "training reference," one with slightly shifted values as "production data from last month" (e.g., shift average order value down 20% to simulate a recession-like behavior change).
3. Run Evidently's `DataDriftPreset` report comparing the two. You'll get a visual report — literally see "yes, this feature drifted."
4. Wire a simple version into your FastAPI app: log every prediction's input features to a file or table, then run the drift report weekly against that log versus your training data.

Once you've done this once, "data drift monitoring" stops being a term you've heard and becomes something you can describe with your own example.

---

## Layer 3 — Experiment tracking at scale (you already have the basics)

You already have MLflow from the earlier roadmap step. The "at scale" part most courses mean here is just: track metrics not only at training time but at *inference* time — log every prediction's input, output, and the model version that made it, somewhere queryable. That log is also exactly what feeds your drift monitoring in Layer 2. So these two layers aren't separate — the same logging table powers both.

---

## How the three layers connect, concretely

```
Scheduled pipeline (Prefect)
    │
    ├── retrains model weekly
    ├── logs run to MLflow (params, metrics, artifact)
    └── pushes to S3 if better than current

Running API
    │
    ├── serves predictions
    ├── logs every (input, output, model_version) to a table
    └── that log feeds:
            ├── Evidently drift report (weekly)
            └── Prometheus/Grafana dashboards (real-time health)
```

This is genuinely the whole picture — not more complicated than that, just more pieces than one endpoint.

---

## Suggested order, given where you are

1. **Evidently AI first** (1 week) — fastest to a real "aha," cheapest to learn, and directly extends the prediction endpoint you already built.
2. **Prefect second** (1–2 weeks) — turns your `train_model.py` into something that runs on its own.
3. **Prometheus + Grafana third** (1–2 weeks) — slightly more infrastructure-heavy, worth doing once you're comfortable with Docker Compose since they run as just two more services in your existing compose file.
4. **DVC** data versioning

Want me to write out the detailed, code-level walkthrough for these three — same depth as the earlier roadmap — starting with Evidently AI on your churn endpoint?

So, the actual full list, plainly
Stuff you have or will finish soon: Python, FastAPI, Docker, PostgreSQL, basic AWS, CI/CD, basic MLflow.
Stuff that's genuinely next (Circle 2, what we already discussed plus one addition): experiment tracking at scale → pipeline orchestration (Prefect/Airflow) → monitoring & drift detection (Evidently, Prometheus/Grafana) → data versioning (DVC) → enough Kubernetes to not be lost in a conversation about it.
Stuff that comes after that, not instead of it (Circle 3): RAG, vector databases, agent orchestration, evals for LLM output — you've already touched these in your portfolio, so this circle is shorter for you than for most people.
Stuff you don't need to think about yet (Circle 4): leadership/specialization tracks — these aren't a learning checklist, they're where you end up after a few years.

Before the tutorial — you deserve a straight answer to the harder question first.

You haven't had "almost no gain" over 10 years. You have a PhD, a master's, published research, and a real teaching career. That's not nothing — that's the part most people trying to break into ML *can't* get no matter how hard they study. What you're missing is narrow and well-defined: the engineering layer. That's months, not years, and you're already most of the way through it.

Will it pay off? Honestly — yes, with real confidence, for a *good* role, and with appropriate humility about *how* good. The Egypt market we found earlier has real demand, and your research background plus this engineering layer puts you ahead of most candidates competing for those same roles, who usually have one or the other, not both. What I can't promise is a specific number or a specific company — markets move, and anyone who guarantees that isn't being honest with you. What I can tell you is that you are not pouring 10 years into nothing. You're one focused stretch away from a combination that's genuinely rare.

Now, the tutorial.

---

# MLOps Deep Layer — Detailed Tutorial
### DVC, Pipelines, Monitoring, and where Kubernetes fits

---

## Why this layer exists at all — the problem it solves

Everything in your roadmap so far solves "how do I get one model behind one working endpoint." That's real and necessary, but it has a quiet weakness: it assumes the world stands still after you deploy. It doesn't. Customer behavior shifts, your training data goes stale, someone retrains the model differently and you can't remember which dataset produced which result, and nobody is watching to notice when predictions slowly get worse. This layer exists specifically to close those four gaps: reproducibility, automation, observability, and scale. We'll take them one at a time, in the order that builds most naturally on what you already have.

---

# PART 1 — DVC (Data Version Control)

### What problem this actually solves, in plain terms

Git is excellent at tracking code, but terrible at tracking data. If you `git add` a 500MB CSV, your repository becomes huge and slow, and worse — Git has no good way of telling you "this model was trained on *this exact version* of this dataset." Six months from now, if someone asks "why did the model from March behave differently than the one from June," you need to be able to answer precisely. Without data versioning, your honest answer is "I don't know, the data changed at some point." That's not an acceptable answer in a company that takes ML seriously.

DVC solves this by doing for data what Git does for code — but cleverly, without actually putting the heavy files into Git itself. Instead, DVC stores a small text pointer file in Git (a few KB) that says "the real data lives at this location and has this exact hash," while the actual data sits in cheap storage like S3. When you check out an old Git commit, DVC can fetch back the exact data file that existed at that point in time.

### Why you need this specifically, given what you're building

Right now, your `train_model.py` reads from a hardcoded dictionary of fifteen rows. In a real system, you'd be pulling from your `orders` and `customers` tables in PostgreSQL, and that data changes every day as new orders come in. Without DVC, if your model's behavior changes next month, you have no way of proving whether that's because you changed the code or because the underlying data changed. With DVC, you can say precisely: "model run #47 was trained on data version `a3f91c`, and here is exactly what that dataset looked like."

### Setting it up on your retail project

```bash
pip install dvc
pip install dvc-s3   # since you're already using AWS S3
```

Initialize DVC inside your existing git repository — it works alongside Git, not instead of it:

```bash
dvc init
git add .dvc .dvcignore
git commit -m "initialize dvc"
```

This creates a `.dvc` folder. DVC is now aware of your project, but hasn't tracked any data yet.

### Exporting your training data as a real file DVC can track

Right now your data lives as a Python dictionary inside `train_model.py`. To version it properly, we extract it into its own file first:

```python
# export_training_data.py
# This script pulls customer data from your actual PostgreSQL database
# and saves it as a CSV — a file DVC can track and version.
# Why separate this from train_model.py? Because the data extraction step
# and the training step are conceptually different stages of a pipeline,
# and DVC pipelines (which we set up in Part 2) work best when each stage
# is its own script with clear inputs and outputs.

import pandas as pd
from sqlalchemy import create_engine
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/retail_db")
engine = create_engine(DATABASE_URL)

# This query computes the same four features your model already uses,
# but pulled live from real order and customer data instead of being
# hand-typed into a dictionary.
query = """
SELECT
    c.id AS customer_id,
    COUNT(o.id) AS total_orders,
    COALESCE(AVG(o.total_amount), 0) AS avg_order_value,
    COALESCE(EXTRACT(DAY FROM NOW() - MAX(o.created_at)), 999) AS days_since_last_order,
    COALESCE(SUM(o.total_amount), 0) AS total_spent
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id
GROUP BY c.id
"""

df = pd.read_sql(query, engine)
os.makedirs("data", exist_ok=True)
df.to_csv("data/training_data.csv", index=False)
print(f"Exported {len(df)} customer rows to data/training_data.csv")
```

Run it, then hand the resulting file to DVC instead of Git:

```bash
python export_training_data.py
dvc add data/training_data.csv
```

This creates `data/training_data.csv.dvc` — a tiny pointer file containing a hash of the real data. *This* small file is what you commit to Git:

```bash
git add data/training_data.csv.dvc data/.gitignore
git commit -m "track training data v1 with dvc"
```

Notice DVC automatically added `data/training_data.csv` to a `.gitignore` it manages — the actual CSV never goes into Git, only its fingerprint does.

### Pushing the actual data to S3

The pointer file in Git is useless without somewhere real for DVC to fetch the actual data from. We point DVC at the same S3 bucket you already set up:

```bash
dvc remote add -d storage s3://your-name-retail-models/dvc-storage
dvc push
```

`dvc push` uploads the real CSV to S3. From now on, anyone who clones your repository can run `dvc pull` and get the exact data file that matches whichever Git commit they have checked out.

### The workflow you'll actually use day to day

When your data changes (new orders came in, you fixed a data quality issue, whatever):

```bash
python export_training_data.py    # regenerate the CSV with fresh data
dvc add data/training_data.csv     # DVC notices the hash changed
git add data/training_data.csv.dvc
git commit -m "update training data — added Q2 orders"
dvc push
```

Now if you ever need to go back: `git checkout <old commit>` followed by `dvc pull` gives you back the *exact* data file that existed at that point, byte for byte. That is the entire value of DVC in one sentence: your data has the same time-travel guarantee your code already has in Git.

---

# PART 2 — Pipeline orchestration (Prefect)

### What problem this actually solves

Right now, getting a new model into production means you, personally, running a sequence of commands by hand: export data, train, evaluate, upload to S3. Every manual step is a place where you might forget something, do it in the wrong order, or simply not have time to do it that week. A pipeline orchestrator turns that sequence into code that runs itself — on a schedule, or triggered by an event — and handles retries and failures sensibly.

### Why "just write a bash script that runs all the steps" isn't enough

You might think: why not just write one script that calls all four steps in order? You can, and for a single linear sequence that mostly works. But real pipelines need things a plain script handles badly: what happens if step 2 fails — should step 3 still run? How do you see, at a glance, which step failed and why, three weeks from now when you're not looking at your terminal? How do you run this on a schedule without setting up your own cron infrastructure? Prefect (and tools like it) give you a dashboard showing every run, every step's success or failure, and built-in retry logic — for not much more code than the plain script would have taken.

### Setting up Prefect

```bash
pip install prefect
```

We convert your training process into a Prefect flow. The key idea: you wrap each logical step in `@task`, and the overall sequence in `@flow`. Prefect then tracks each task's execution, timing, and success/failure automatically.

```python
# ml_pipeline.py
from prefect import flow, task
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sqlalchemy import create_engine
import joblib
import mlflow
import boto3
import os
from loguru import logger


@task(retries=2, retry_delay_seconds=10)
def extract_data() -> pd.DataFrame:
    """
    Pull fresh customer data from PostgreSQL.

    retries=2 means: if this fails (e.g. the database is briefly
    unreachable), Prefect automatically tries again twice before
    giving up, waiting 10 seconds between attempts. This is exactly
    the kind of resilience a plain script doesn't give you for free.
    """
    database_url = os.getenv("DATABASE_URL")
    engine = create_engine(database_url)
    query = """
        SELECT c.id AS customer_id,
               COUNT(o.id) AS total_orders,
               COALESCE(AVG(o.total_amount), 0) AS avg_order_value,
               COALESCE(EXTRACT(DAY FROM NOW() - MAX(o.created_at)), 999) AS days_since_last_order,
               COALESCE(SUM(o.total_amount), 0) AS total_spent
        FROM customers c
        LEFT JOIN orders o ON o.customer_id = c.id
        GROUP BY c.id
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Extracted {len(df)} rows")
    return df


@task
def train_model(df: pd.DataFrame):
    """
    Train the churn model and log everything to MLflow.
    This task's only job is training and logging — it doesn't decide
    whether the model is good enough to deploy. That decision belongs
    to a separate task, which keeps each task focused on one concern.
    """
    # In your real data you'd have a 'churned' label column already;
    # here we assume it exists from your business logic.
    X = df[["total_orders", "avg_order_value", "days_since_last_order", "total_spent"]]
    y = df["churned"]  # however you define/derive this label in your data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

    logger.info(f"Trained model. Accuracy={accuracy:.3f}, F1={f1:.3f}, run_id={run_id}")
    return {"model": model, "accuracy": accuracy, "f1": f1, "run_id": run_id}


@task
def evaluate_against_production(new_metrics: dict) -> bool:
    """
    Decide whether the new model is actually good enough to replace
    the one currently in production. This is the step that prevents
    a pipeline from blindly overwriting a good model with a worse one
    just because retraining happened on schedule.

    For now we use a simple hardcoded baseline; in a real system you'd
    fetch the current production model's recorded accuracy from MLflow
    or a small metadata file and compare against that.
    """
    CURRENT_PRODUCTION_ACCURACY = 0.85
    is_better = new_metrics["accuracy"] >= CURRENT_PRODUCTION_ACCURACY
    logger.info(
        f"New accuracy: {new_metrics['accuracy']:.3f}, "
        f"current production: {CURRENT_PRODUCTION_ACCURACY}, "
        f"promote: {is_better}"
    )
    return is_better


@task
def deploy_to_s3(model, should_deploy: bool):
    """
    Only push to S3 if the evaluation task approved it.
    This guard is what makes "automatic retraining" safe rather than
    reckless — automation without a quality gate is how you accidentally
    ship a worse model to production on autopilot.
    """
    if not should_deploy:
        logger.warning("New model did not beat production baseline. Skipping deploy.")
        return

    joblib.dump(model, "models/churn_model.pkl")
    s3 = boto3.client("s3")
    s3.upload_file("models/churn_model.pkl", "your-name-retail-models", "churn_model.pkl")
    logger.info("New model deployed to S3.")


@flow(name="churn-model-retraining")
def churn_retraining_pipeline():
    """
    This is the orchestrating flow — it defines the order tasks run in
    and passes data between them. Prefect automatically builds a
    dependency graph from how you call these functions: train_model
    depends on extract_data's output, evaluate depends on train_model's
    output, and so on.
    """
    df = extract_data()
    result = train_model(df)
    should_deploy = evaluate_against_production(result)
    deploy_to_s3(result["model"], should_deploy)


if __name__ == "__main__":
    churn_retraining_pipeline()
```

Run it directly to confirm it works:

```bash
python ml_pipeline.py
```

### Turning this into something that runs on a schedule, automatically

Running the script by hand is still manual. To make it actually automatic, you deploy it with a schedule:

```bash
prefect deployment build ml_pipeline.py:churn_retraining_pipeline \
    --name "weekly-churn-retrain" \
    --cron "0 2 * * 1"   # every Monday at 2am
```

```bash
prefect deployment apply churn_retraining_pipeline-deployment.yaml
```

Then start Prefect's local server to see it in a dashboard:

```bash
prefect server start
```

Open `http://localhost:4200` — you will see your flow, its scheduled runs, and (once it has run) full logs of exactly which task succeeded, which failed, and how long each took. This dashboard view is the actual deliverable of orchestration: visibility into a process that used to live only in your head and your terminal history.

---

# PART 3 — Monitoring and drift detection (Evidently AI)

### What problem this actually solves, and why it's different from normal monitoring

If your API crashes, you know immediately — you get an error, a 500 status code, an alert. Drift is the opposite kind of problem: the model keeps running, keeps returning confident-looking predictions, and never throws an error — it just becomes *wrong* more often, silently, over weeks or months. This happens because the relationship between customer behavior and churn that the model learned slowly stops matching reality. Maybe a new competitor enters the market, maybe a pricing change shifts what "normal" order value looks like. The model has no way of knowing this on its own — it just keeps confidently applying old patterns to new reality.

Monitoring for drift means continuously comparing the data the model is *currently* seeing against the data it was *trained* on, and flagging when they've grown meaningfully different.

### Setting up Evidently AI

```bash
pip install evidently
```

### Step one: actually capture what the model sees in production

Before you can detect drift, you need a record of every prediction request that came in. Update your prediction router to log this:

```python
import csv
from datetime import datetime

PREDICTION_LOG_PATH = "logs/prediction_log.csv"

def log_prediction(input_data: ChurnPredictionInput, output: ChurnPredictionOutput):
    """
    Append every prediction's input and output to a CSV log.
    This log is the raw material for drift detection — without it,
    you have nothing to compare against your training data.
    """
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(PREDICTION_LOG_PATH)

    with open(PREDICTION_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "total_orders", "avg_order_value",
                "days_since_last_order", "total_spent",
                "will_churn", "churn_probability",
            ])
        writer.writerow([
            datetime.utcnow().isoformat(),
            input_data.total_orders, input_data.avg_order_value,
            input_data.days_since_last_order, input_data.total_spent,
            output.will_churn, output.churn_probability,
        ])
```

Call `log_prediction(input_data, result)` right before `return result` in your `/predict/churn` endpoint.

### Step two: generate a drift report comparing training data to production logs

```python
# check_drift.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# "Reference" is the data the model was trained on — the baseline of
# "normal" that we're comparing everything else against.
reference_data = pd.read_csv("data/training_data.csv")
reference_data = reference_data[["total_orders", "avg_order_value", "days_since_last_order", "total_spent"]]

# "Current" is what the model has actually been seeing in production
# recently — pulled from the log file we just started writing.
current_data = pd.read_csv("logs/prediction_log.csv")
current_data = current_data[["total_orders", "avg_order_value", "days_since_last_order", "total_spent"]]

# DataDriftPreset runs a battery of statistical tests (one per column)
# comparing the distribution of each feature between reference and
# current data, and summarizes whether each one has drifted.
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

report.save_html("drift_report.html")
print("Drift report saved to drift_report.html")

# You can also pull the result as a Python dict to act on programmatically —
# for example, to trigger an alert or even kick off the retraining pipeline
# from Part 2 if drift is severe enough.
result = report.as_dict()
dataset_drift_detected = result["metrics"][0]["result"]["dataset_drift"]
print(f"Overall dataset drift detected: {dataset_drift_detected}")
```

Run it and open `drift_report.html` in a browser. You'll see, feature by feature, whether the distribution of values your model is *currently seeing* has meaningfully shifted away from what it was *trained on* — with actual visual histograms comparing the two, not just a yes/no.

### Why this connects back to Part 2

Notice the last line of that script — `dataset_drift_detected`. This is exactly the kind of signal that should feed back into your Prefect pipeline: instead of *only* retraining on a fixed weekly schedule, you can add a check — "has drift been detected since the last retrain?" — and only bother retraining when it's actually warranted. This is what people mean when they talk about pipelines and monitoring "working together" rather than being separate concerns.

### A note on infrastructure-level monitoring (Prometheus + Grafana)

Evidently answers "has my model's *world* changed." A different question — "is my *API* healthy right now" (is it slow? erroring? running out of memory?) — is what Prometheus and Grafana answer. You don't need to build this immediately, but it's worth knowing the distinction: Evidently is about the model's correctness over time; Prometheus/Grafana is about the service's operational health right now. Both matter, but if you only have time for one right now, Evidently is the one that's specifically about machine learning, and the one most directly tied to your churn model's actual usefulness.

---

# PART 4 — Where Kubernetes fits, and how much you actually need to know

### The honest scope of what you need here, right now

You do not need to become a Kubernetes expert. You need to understand *what problem it solves* and recognize the handful of concepts that come up constantly in interviews and job descriptions, so you're never lost in a conversation about it. Real depth here can come later, if a job actually requires it.

### What problem Kubernetes solves, in plain terms

Docker Compose, which you already know, is excellent for running your stack on *one machine*. But what happens when one server isn't enough — when your `/predict/churn` endpoint needs to handle far more traffic than one container can manage, or when a server crashes at 3am and someone needs to notice and recover automatically? Kubernetes is the tool for running containers across *many* machines, automatically restarting ones that crash, automatically adding more copies of your app when traffic increases, and automatically routing requests to whichever copies are healthy.

### The core concepts, explained simply

A **Pod** is the smallest unit Kubernetes manages — essentially one or more containers running together, similar in spirit to one service in your Docker Compose file. A **Deployment** describes how many copies (replicas) of a Pod should be running at all times — if you say "I want 3 replicas of my API," Kubernetes continuously works to keep exactly 3 healthy copies running, restarting any that crash. A **Service** is a stable network address that routes traffic to whichever Pods are currently healthy, so the rest of your system doesn't need to know or care which specific Pod handles a given request. A **ConfigMap** and **Secret** are how you inject configuration and credentials into Pods — conceptually similar to the `environment:` section in your Compose file, just managed by Kubernetes instead.

### A minimal example, just to see the shape of it

You don't need to run this anywhere yet — just read it to see how it maps to what you already know from Compose:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retail-api
spec:
  replicas: 3   # Kubernetes keeps 3 copies running at all times
  selector:
    matchLabels:
      app: retail-api
  template:
    metadata:
      labels:
        app: retail-api
    spec:
      containers:
        - name: retail-api
          image: yourdockerhubusername/retail-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: retail-secrets
                  key: database-url
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: retail-api-service
spec:
  selector:
    app: retail-api
  ports:
    - port: 80
      targetPort: 8000
```

Compare this mentally to your `docker-compose.yml` `api` service — the `image`, `ports`, and `environment` sections map almost directly. The genuinely new ideas are `replicas` (running multiple copies) and the `Service` acting as a stable front door to whichever copies are currently alive.

### What's actually worth doing, practically, right now

Install `minikube` (a single-node Kubernetes cluster that runs on your laptop) and follow its official "deploy your first app" tutorial using your own retail API image instead of their example. That's genuinely enough for now — being able to say in an interview "I've run my own container on a local Kubernetes cluster and understand Deployments and Services" is a real, honest, sufficient claim at your current career stage. Going deeper (Helm charts, autoscaling, multi-node clusters) is a Circle-2-going-into-Circle-3 skill, not something blocking your first roles.

---

## Where this leaves you

DVC gives you reproducibility — you can always answer "what data made this model." Prefect gives you automation — retraining happens on its own, safely gated by an evaluation step. Evidently gives you truth — you'll know when the world has changed under your model's feet, instead of finding out from a confused stakeholder three months later. And a light grasp of Kubernetes means you're never lost when a job description or an interviewer brings it up, even if you're not running production clusters yet.

That's the complete Circle 2. Once you've built even small versions of all four — and you genuinely can, on this same retail project — you are, without exaggeration, further along than most candidates with a "Machine Learning Engineer" job title already on their CV.