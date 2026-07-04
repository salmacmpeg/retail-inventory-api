from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException
from loguru import logger
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from RetailApp.database import get_db
from RetailApp.dependencies import get_current_admin, get_current_customer
from RetailApp.models import Order as OrderTable
from RetailApp.models import OrderItem as OrderItemTable
from RetailApp.models import Product as ProductTable
from RetailApp.models import User as UserTable
from RetailApp.routes.services import check_customer_exists, find_order
from RetailApp.schemas import Order, OrderIN, status_choice

router = APIRouter()


@router.post("/order", response_model=Order, status_code=201)
async def add_order(
    order_data: OrderIN, db: AsyncSession = Depends(get_db), current_customer: UserTable = Depends(get_current_customer)
):
    customer_id = current_customer.id

    logger.debug(f"Attempting to create order for customer_id: {customer_id}")

    # Validation Logic
    if not await check_customer_exists(customer_id, db):
        logger.warning(f"Order creation failed: Customer {customer_id} does not exist")
        raise HTTPException(status_code=404, detail="Customer not found")

    # Database Logic
    try:
        order_total = 0
        items_to_create = []

        # validating for every item in the list
        for cart_item in order_data.cart_items:
            sql_query = select(ProductTable).where(ProductTable.id == cart_item.item_id)
            result = await db.execute(sql_query)
            product = result.scalar_one_or_none()

            if not product:
                raise HTTPException(status_code=404, detail=f"Product with ID {cart_item.item_id} doesnt exist")

            if product.inventory_level - cart_item.quantity < 0:
                raise HTTPException(
                    status_code=400, detail=f"Not enough stock for this item with ID {cart_item.item_id}"
                )

            item_cost = product.current_price * cart_item.quantity
            order_total += item_cost
            product.inventory_level -= cart_item.quantity

            items_to_create.append(
                {
                    "item_id": cart_item.item_id,
                    "quantity": cart_item.quantity,
                    "price_at_purchase": product.current_price,
                }
            )

        created_at = datetime.now(timezone.utc)
        statement = insert(OrderTable).values(
            customer_id=customer_id,
            order_total=order_total,
            status=order_data.status,
            created_at=created_at,
        )
        logger.debug("Executing database insert for new order")
        result = await db.execute(statement)
        order_id = result.lastrowid

        bulk_insertion = []

        for row_data in items_to_create:
            bulk_insertion.append(
                {
                    "order_id": order_id,
                    "item_id": row_data["item_id"],
                    "quantity": row_data["quantity"],
                    "price_at_purchase": row_data["price_at_purchase"],
                }
            )

        await db.execute(insert(OrderItemTable).values(bulk_insertion))

        await db.commit()

        db.expire_all()

        final_query = (
            select(OrderTable)
            .where(OrderTable.id == order_id)
            .options(joinedload(OrderTable.items))
            .execution_options(populate_existing=True)
        )

        raw_result = await db.execute(final_query)
        final_result = raw_result.unique()

        logger.info(f"Order created successfully: ID {order_id} for Customer {customer_id}")

        return final_result.scalars().unique().one()

    except HTTPException:
        await db.rollback()
        raise

    except Exception as e:
        # Unexpected DB errors get the ERROR level
        await db.rollback()
        logger.error(f"Unexpected database error during order creation: {str(e)}")
        raise


@router.get("/order/{id}", response_model=Order, status_code=201)
async def get_order(
    id: int, db: AsyncSession = Depends(get_db), current_customer: UserTable = Depends(get_current_customer)
):
    logger.debug(f"Fetching order details for ID: {id}")

    order = await find_order(id, db)

    if not order:
        logger.warning(f"Lookup failed: Order ID {id} not found")
        raise HTTPException(status_code=404, detail="Order not found")

    if order.customer_id != current_customer.id:
        logger.warning(f"Unauthorized access attempt by user {current_customer.id} for order ID {id}")
        raise HTTPException(status_code=403, detail="You do not have permission to view this order")

    logger.info(f"Order {id} retrieved successfully")
    return order


@router.get("/orders/me", response_model=list[Order], status_code=201)
async def get_my_orders(
    db: AsyncSession = Depends(get_db), current_customer: UserTable = Depends(get_current_customer)
):
    logger.debug("Fetching all orders from database")

    statement = (
        select(OrderTable).options(joinedload(OrderTable.items)).where(OrderTable.customer_id == current_customer.id)
    )
    results = await db.execute(statement)
    orders = results.scalars().unique().all()

    logger.info(f"Successfully retrieved {len(orders)} orders")
    return list(orders)


@router.get("/orders", response_model=list[Order], status_code=201)
async def get_all_orders(db: AsyncSession = Depends(get_db), current_admin: UserTable = Depends(get_current_admin)):
    logger.debug("Fetching all orders from database")

    statement = select(OrderTable).options(joinedload(OrderTable.items))
    results = await db.execute(statement)
    orders = results.scalars().unique().all()

    logger.info(f"Successfully retrieved {len(orders)} orders")
    return list(orders)


@router.patch("/order/{order_id}/status", response_model=Order, status_code=201)
async def update_order_status(
    order_id: int,
    new_status: status_choice = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_admin: UserTable = Depends(get_current_admin),
):
    logger.debug(f"Attempting to Modify Order with ID {order_id} Status Successfully to {new_status}")
    logger.debug(f"Fetching order details for ID: {id}")

    statement = select(OrderTable).options(joinedload(OrderTable.items)).where(OrderTable.id == order_id)
    result = await db.execute(statement)
    order = result.scalars().unique().one_or_none()

    if not order:
        logger.warning(f"Lookup failed: Order ID {order_id} not found")
        raise HTTPException(status_code=404, detail="Order not found")

    order.status = new_status
    await db.commit()

    logger.info(f"Modified Order with ID {order_id} Status Successfully to {new_status}")

    db.expire_all()

    result = await db.execute(statement)
    order = result.scalars().unique().one_or_none()
    return order
