from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from RetailApp.database import get_db
from RetailApp.dependencies import get_current_admin, get_current_customer, get_current_user
from RetailApp.models import Product as ProductTable
from RetailApp.models import User as UserTable
from RetailApp.routes.services import find_product, get_products
from RetailApp.schemas import Product, ProductIN

router = APIRouter()


@router.post("/product", response_model=Product, status_code=201)
async def add_product(
    product: ProductIN, db: AsyncSession = Depends(get_db), current_admin: UserTable = Depends(get_current_admin)
):
    data = product.model_dump()
    logger.debug(f"Attempting to create product with data : {data}")

    now_utc = datetime.now(timezone.utc)
    file_suffix = now_utc.strftime("%y%m%d_%H%M%S")

    try:
        sku = data["name"] + "_" + data["category"] + "_" + str(file_suffix)
        statement = insert(ProductTable).values(**data, sku=sku)

        logger.debug("Executing database insert for new product")
        result = await db.execute(statement)
        await db.commit()

        # new_product = result.scalar_one()
        new_product_id = result.lastrowid

        logger.info(f"Product created successfully: ID {new_product_id} for admin with user id = {current_admin.id}")

        statement = select(ProductTable).where(ProductTable.id == new_product_id)
        result = await db.execute(statement)
        new_product = result.scalar_one()

        return new_product

    except Exception as e:
        # Unexpected DB errors get the ERROR level
        logger.error(f"Unexpected database error during product creation: {str(e)}")
        raise


@router.get("/products", response_model=list[Product], status_code=201)
async def get_all_products(db: AsyncSession = Depends(get_db), current_customer: UserTable = Depends(get_current_user)):
    logger.debug("Fetching all products from database")

    if current_customer.role == "admin":
        products = await get_products(db)

    else:
        statement = select(ProductTable).where(ProductTable.inventory_level > 0)
        results = await db.execute(statement)
        products = list(results.scalars().all())

    logger.info(f"Successfully retrieved {len(products)} products")
    return products


@router.get("/product/{id}", response_model=Product, status_code=201)
async def get_product(
    id: int, db: AsyncSession = Depends(get_db), current_customer: UserTable = Depends(get_current_user)
):
    logger.debug(f"Fetching product details for ID: {id}")

    product = await find_product(id, db)

    if not product:
        logger.warning(f"Lookup failed: Product ID {id} not found")
        raise HTTPException(status_code=404, detail="Product not found")

    logger.info(f"Product {id} retrieved successfully")
    return product
