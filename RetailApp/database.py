from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from RetailApp.core.config import config

# database url
# async engine
# session maker
# base class
# database tables
# get db session

engine = create_async_engine(
    config.DATABASE_URL,
    echo=True,
    future=True,
)

async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as db:
        print("********************************Opening DB session")
        yield db
    print("********************************Closing DB session")


# async def get_db():
#     async with async_session() as session:
#         yield session
#         # Use your config setting to determine if you should roll back
#         if config.DB_FORCE_ROLL_BACK:
#             await session.rollback()
#         else:
#             await session.commit()
