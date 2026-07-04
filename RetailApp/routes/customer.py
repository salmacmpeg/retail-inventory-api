from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from RetailApp.database import get_db
from RetailApp.dependencies import get_current_admin
from RetailApp.models import Customer as CustomerTable
from RetailApp.routes.services import find_customer
from RetailApp.schemas import Customer, CustomerOut

router = APIRouter()


@router.get("/customer/{id}", response_model=CustomerOut, status_code=201)
async def get_customer(
    id: int, db: AsyncSession = Depends(get_db), current_admin: CustomerTable = Depends(get_current_admin)
):
    logger.debug(f"Searching for customer ID: {id}")
    customer = await find_customer(id, db)

    if not customer:
        logger.warning(f"Customer lookup failed: ID {id} not found")
        raise HTTPException(status_code=404, detail="Customer not found")

    logger.info(f"Customer {id} found")
    return customer


@router.get("/customers", response_model=list[Customer], status_code=201)
async def get_customers(db: AsyncSession = Depends(get_db), current_admin: CustomerTable = Depends(get_current_admin)):
    logger.debug("Querying all customers")
    statement = select(CustomerTable)
    results = await db.execute(statement)
    customers = results.scalars().all()

    logger.info(f"Retrieved {len(customers)} customers from database")
    return list(customers)
