from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from jose import JWTError
from loguru import logger
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from RetailApp.core.config import config
from RetailApp.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from RetailApp.database import get_db
from RetailApp.models import Customer as CustomerTable
from RetailApp.models import RefreshToken as RefreshTokenTable
from RetailApp.models import User as UserTable
from RetailApp.schemas import RefreshTokenRequest, TokenResponse, UserCreate, UserLogin, UserResponse

router = APIRouter()


@router.post("/user", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    existing_user_query = select(UserTable).where(UserTable.useremail == user.useremail)
    existing_user_result = await db.execute(existing_user_query)
    existing_user = existing_user_result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=409, detail="A user with this email already exists.")

    hashed_password = hash_password(user.password)
    new_user = {
        "username": user.username,
        "useremail": user.useremail,
        "userphone": user.userphone,
        "hashed_password": hashed_password,
        "is_active": True,
        "role": user.role,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        sql_query = insert(UserTable).values(**new_user)
        result = await db.execute(sql_query)
        await db.commit()

        new_user_id = result.lastrowid

        sql_query = select(UserTable).where(UserTable.id == new_user_id)
        result = await db.execute(sql_query)
        new_user = result.scalar_one()

        logger.info(f"User created successfully: ID {new_user.id} | Email: {new_user.useremail}")

        if user.role == "customer":
            customer_data = {
                "name": user.username,
                "email": user.useremail,
                "phone": user.userphone,
                "user_id": new_user.id,
            }
            customer_query = insert(CustomerTable).values(**customer_data)
            customer_result = await db.execute(customer_query)

            await db.commit()

            new_customer_id = customer_result.lastrowid
            new_user.customer_id = new_customer_id

            logger.info(f"Customer profile with ID {new_customer_id} created successfully for user ID {new_user.id}")

        return new_user

    except Exception as e:
        await db.rollback()
        logger.error(f"Error occurred while creating user: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while creating the user.")


@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    user_mail = user.useremail

    sql_query = select(UserTable).where(UserTable.useremail == user_mail)
    result = await db.execute(sql_query)
    db_user = result.scalar_one_or_none()

    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not db_user.is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    access_token = create_access_token(data={"sub": str(db_user.id)})
    refresh_token = create_refresh_token(data={"sub": str(db_user.id)})

    expire = datetime.now(timezone.utc) + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)

    sql_query = insert(RefreshTokenTable).values(
        user_id=db_user.id, token=refresh_token, expires_at=expire, is_revoked=False
    )
    await db.execute(sql_query)
    await db.commit()

    logger.info(f"User logged in successfully: ID {db_user.id} | Email: {db_user.useremail}")

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Invalid refresh token")

    try:
        token_data = decode_token(refresh_token.refresh_token)
        if token_data.get("type") != "refresh" or token_data.get("sub") is None:
            raise credentials_exception
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")

    sql_query = select(RefreshTokenTable).where(RefreshTokenTable.token == refresh_token.refresh_token)
    result = await db.execute(sql_query)
    db_token = result.scalar_one_or_none()
    if db_token is None or db_token.is_revoked:
        raise credentials_exception

    db_token.is_revoked = True
    await db.commit()

    access_token = create_access_token(data={"sub": str(db_token.user_id)})
    new_refresh_token = create_refresh_token(data={"sub": str(db_token.user_id)})

    expire = datetime.now(timezone.utc) + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)
    sql_query = insert(RefreshTokenTable).values(
        user_id=db_token.user_id, token=new_refresh_token, expires_at=expire, is_revoked=False
    )
    await db.execute(sql_query)
    await db.commit()

    logger.info(f"Refresh token used successfully: User ID {db_token.user_id}")

    return TokenResponse(access_token=access_token, refresh_token=new_refresh_token)


@router.post("/logout", status_code=204)
async def logout(refresh_token: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    sql_query = select(RefreshTokenTable).where(RefreshTokenTable.token == refresh_token.refresh_token)
    result = await db.execute(sql_query)
    db_token = result.scalar_one_or_none()
    if db_token:
        db_token.is_revoked = True
        await db.commit()
