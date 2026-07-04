import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from RetailApp.core.config import config
from RetailApp.core.logging import setup_logging  # critical, error, warning, info, debug
from RetailApp.database import Base, engine
from RetailApp.routes.authenticate import router as auth_router
from RetailApp.routes.customer import router as customer_router
from RetailApp.routes.order import router as order_router
from RetailApp.routes.pricer import load_ml_assets
from RetailApp.routes.pricer import router as pricer_router
from RetailApp.routes.product import router as products_router
from RetailApp.routes.users import router as users_router

setup_logging()


@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    logger.info("********************************App startup")
    logger.info(f"--- ACTIVE CONFIG: {type(config).__name__} ---")
    logger.info(f"--- DATABASE: {config.DATABASE_URL} ---")
    logger.info(f"--- DB_FORCE_ROLL_BACK: {config.DB_FORCE_ROLL_BACK} ---")
    logger.info(f"--- LOG_LEVEL: {config.LOG_LEVEL} ---")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.state.ml_model, app.state.category_anchors = load_ml_assets()
    yield
    await engine.dispose()
    logger.info("********************************App shutdown")


app = FastAPI(lifespan=lifespan_handler, title="Retail Inventory API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,  # your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.middleware("http")
async def logging_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    with logger.contextualize(request_id=request_id):
        logger.info(f"Starting request: {request_id} | {request.method} {request.url.path} from {request.client.host}")
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                f"Completed request: {request_id} | {request.method}"
                f" {request.url.path} from {request.client.host} | Status code: "
                f"{response.status_code} | Process time: {process_time:.3f}s"
            )
            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.exception(
                f"Request failed: {request_id} {request.method} {request.url.path}|"
                f" from {request.client.host} | Exception: {e} | Process time: {process_time:.3f}s"
            )
            raise


app.include_router(customer_router)
app.include_router(order_router)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(products_router)
app.include_router(pricer_router)


@app.get("/")
async def read_login():
    with open("frontend/pages/login.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)
