from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select

from RetailApp.database import get_db
from RetailApp.dependencies import get_current_admin, get_current_user
from RetailApp.models import Customer as CustomerTable
from RetailApp.models import RefreshToken as RefreshTokenTable
from RetailApp.models import User as UserTable
from RetailApp.schemas import UserResponse

router = APIRouter()


@router.get("/users/me", response_model=UserResponse, status_code=200)
async def read_current_user(current_user: UserTable = Depends(get_current_user)):
    logger.info(f"Fetching current user: ID {current_user.id} | Email: {current_user.useremail}")
    if current_user.is_active is False:
        raise HTTPException(status_code=403, detail="User is Deactivated")
    return current_user


@router.delete("/users/me", status_code=204)
async def delete_current_user(current_user: UserTable = Depends(get_current_user), db=Depends(get_db)):
    logger.info(f"Deleting current user: ID {current_user.id} | Email: {current_user.useremail}")

    refresh_token_query = select(RefreshTokenTable).where(RefreshTokenTable.user_id == current_user.id)
    result = await db.execute(refresh_token_query)
    refresh_tokens = result.scalars().all()
    for token in refresh_tokens:
        await db.delete(token)

    if current_user.role == "customer":
        customer_query = select(CustomerTable).where(CustomerTable.user_id == current_user.id)
        result = await db.execute(customer_query)
        customer = result.scalar_one_or_none()
        if customer:
            await db.delete(customer)

    await db.delete(current_user)
    await db.commit()
    return {"detail": "User deleted successfully"}


@router.get("/users/list", response_model=list[UserResponse], status_code=200)
async def read_all_users(current_admin: UserTable = Depends(get_current_admin), db=Depends(get_db)):
    logger.info(f"Fetching all users by admin: ID {current_admin.id} | Email: {current_admin.useremail}")
    sql_query = select(UserTable)
    result = await db.execute(sql_query)
    users = result.scalars().all()
    return users


@router.put("/users/{user_id}/deactivate", response_model=UserResponse, status_code=200)
async def deactivate_user(user_id: int, current_admin: UserTable = Depends(get_current_admin), db=Depends(get_db)):
    logger.info(f"Deactivating user: ID {user_id} by admin: ID {current_admin.id} | Email: {current_admin.useremail}")
    sql_query = select(UserTable).where(UserTable.id == user_id)
    result = await db.execute(sql_query)
    user_to_deactivate = result.scalar_one_or_none()

    if not user_to_deactivate:
        raise HTTPException(status_code=404, detail="User not found")

    user_to_deactivate.is_active = False

    db.add(user_to_deactivate)
    await db.commit()
    await db.refresh(user_to_deactivate)
    return user_to_deactivate


@router.put("/users/{user_id}/activate", status_code=200)
async def activate_user(user_id: int, current_admin: UserTable = Depends(get_current_admin), db=Depends(get_db)):
    logger.info(f"Activating user: ID {user_id} by admin: ID {current_admin.id} | Email: {current_admin.useremail}")
    sql_query = select(UserTable).where(UserTable.id == user_id)
    result = await db.execute(sql_query)
    user_to_activate = result.scalar_one_or_none()

    if not user_to_activate:
        raise HTTPException(status_code=404, detail="User not found")

    user_to_activate.is_active = True

    db.add(user_to_activate)
    await db.commit()
    await db.refresh(user_to_activate)
    return user_to_activate
