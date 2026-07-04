from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from RetailApp.models import Customer as CustomerTable
from RetailApp.models import Order as OrderTable
from RetailApp.models import Product as ProductTable
from RetailApp.schemas import CustomerOut, Order


async def check_customer_exists(customer_id: int, db: AsyncSession) -> bool:
    statement = select(CustomerTable).where(CustomerTable.id == customer_id)
    result = await db.execute(statement)
    customer = result.scalar_one_or_none()
    return customer is not None


async def find_customer(customer_id: int, db: AsyncSession) -> CustomerOut | None:
    statement = select(CustomerTable).where(CustomerTable.id == customer_id)
    result = await db.execute(statement)
    customer = result.scalar_one_or_none()
    return customer


async def find_order(order_id: int, db: AsyncSession) -> Order | None:
    statement = select(OrderTable).options(joinedload(OrderTable.items)).where(OrderTable.id == order_id)
    result = await db.execute(statement)
    order = result.unique().scalar_one_or_none()
    return order


async def find_product(product_id: int, db: AsyncSession) -> ProductTable | None:
    statement = select(ProductTable).where(ProductTable.id == product_id)
    result = await db.execute(statement)
    product = result.scalar_one_or_none()
    return product


async def get_orders(customer_id: int, db: AsyncSession):
    # Logic to retrieve all orders from the database
    statement = select(OrderTable).where(OrderTable.customer_id == customer_id)
    results = await db.execute(statement)
    orders = results.scalars().all()
    return list(orders)


async def get_products(db: AsyncSession):
    # Logic to retrieve all orders from the database
    statement = select(ProductTable)
    results = await db.execute(statement)
    products = results.scalars().all()
    return list(products)
