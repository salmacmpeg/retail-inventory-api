from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from RetailApp.core.security import decode_token
from RetailApp.database import get_db
from RetailApp.models import Customer as CustomerTable
from RetailApp.models import User as UserTable

bearer_scheme = HTTPBearer()


async def get_current_user(
    credintials: HTTPAuthorizationCredentials = Depends(bearer_scheme), db: AsyncSession = Depends(get_db)
) -> UserTable:
    token = credintials.credentials
    try:
        payload = decode_token(token)
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

        sql_query = select(UserTable).where(UserTable.id == user_id)
        result = await db.execute(sql_query)
        db_user = result.scalar_one_or_none()

        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        if not db_user.is_active:
            raise HTTPException(status_code=403, detail="Account is deactivated")

        return db_user

    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")


async def get_current_admin(current_user: UserTable = Depends(get_current_user)) -> UserTable:
    if not current_user.role == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You do not have permission to access this resource"
        )
    return current_user


async def get_current_customer(
    current_user: UserTable = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> CustomerTable:
    if current_user.role != "customer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You do not have permission to access this resource"
        )

    sql_query = select(CustomerTable).where(CustomerTable.user_id == current_user.id)
    result = await db.execute(sql_query)
    db_customer = result.scalar_one_or_none()

    if not db_customer:
        raise HTTPException(status_code=404, detail="Customer profile not found for the current user")

    return db_customer
