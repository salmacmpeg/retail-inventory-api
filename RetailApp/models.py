import sqlalchemy
from sqlalchemy.orm import relationship

from RetailApp.database import Base


class User(Base):
    __tablename__ = "user"
    id = sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True)
    username = sqlalchemy.Column("username", sqlalchemy.String(50), nullable=False, unique=True)
    useremail = sqlalchemy.Column("email", sqlalchemy.String(254), nullable=False, unique=True)
    userphone = sqlalchemy.Column("phone", sqlalchemy.String(20), nullable=False, unique=True)
    hashed_password = sqlalchemy.Column("hashed_password", sqlalchemy.String(255), nullable=False)
    is_active = sqlalchemy.Column("is_active", sqlalchemy.Boolean, nullable=False)
    role = sqlalchemy.Column("role", sqlalchemy.String(30), nullable=False)
    created_at = sqlalchemy.Column("created_at", sqlalchemy.DateTime, nullable=False)

    customer_profile = relationship("Customer", back_populates="user", uselist=False)
    refresh_tokens = relationship("RefreshToken", back_populates="user")


class Customer(Base):
    __tablename__ = "customer"
    id = sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True)
    name = sqlalchemy.Column("name", sqlalchemy.String(100), nullable=False)
    email = sqlalchemy.Column("email", sqlalchemy.String(254), nullable=False, unique=True)
    phone = sqlalchemy.Column("phone", sqlalchemy.String(20), nullable=False, unique=True)
    user_id = sqlalchemy.Column(
        "user_id",
        sqlalchemy.ForeignKey("user.id"),
        nullable=False,
    )

    user = relationship("User", back_populates="customer_profile")
    orders = relationship("Order", back_populates="customer")


class Order(Base):
    __tablename__ = "order"
    id = sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True)
    customer_id = sqlalchemy.Column(
        "customer_id",
        sqlalchemy.ForeignKey("customer.id"),
        nullable=False,
    )
    order_total = sqlalchemy.Column("order_total", sqlalchemy.Float, nullable=False)
    status = sqlalchemy.Column("status", sqlalchemy.String(20), nullable=False)
    created_at = sqlalchemy.Column("created_at", sqlalchemy.DateTime, nullable=False)

    items = relationship("OrderItem", back_populates="parent_order")
    customer = relationship("Customer", back_populates="orders")


class Product(Base):
    __tablename__ = "product"
    id = sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True)
    name = sqlalchemy.Column("name", sqlalchemy.String(150), nullable=False)
    sku = sqlalchemy.Column(
        "sku", sqlalchemy.String(255), nullable=False, unique=True
    )  # Stock Keeping Unit is a unique alphanumeric code assigned to products by retailers
    category = sqlalchemy.Column("category", sqlalchemy.String(50), nullable=False)
    cost_price = sqlalchemy.Column("cost_price", sqlalchemy.Float, nullable=False)
    current_price = sqlalchemy.Column("current_price", sqlalchemy.Float, nullable=False)
    inventory_level = sqlalchemy.Column("inventory_level", sqlalchemy.Integer, default=0, nullable=False)

    ordered_item = relationship("OrderItem", back_populates="product")


class OrderItem(Base):
    __tablename__ = "order_item"
    id = sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True)
    order_id = sqlalchemy.Column(
        "order_id",
        sqlalchemy.ForeignKey("order.id"),
        nullable=False,
    )
    item_id = sqlalchemy.Column(
        "item_id",
        sqlalchemy.ForeignKey("product.id"),
        nullable=False,
    )
    quantity = sqlalchemy.Column("quantity", sqlalchemy.Integer, nullable=False)
    price_at_purchase = sqlalchemy.Column("price_at_purchase", sqlalchemy.Float, nullable=False)

    parent_order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="ordered_item")


class RefreshToken(Base):
    __tablename__ = "refresh_token"
    id = sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True)
    user_id = sqlalchemy.Column(
        "user_id",
        sqlalchemy.ForeignKey("user.id"),
        nullable=False,
    )
    token = sqlalchemy.Column("token", sqlalchemy.String(512), nullable=False, unique=True)
    expires_at = sqlalchemy.Column("expires_at", sqlalchemy.DateTime, nullable=False)
    is_revoked = sqlalchemy.Column("is_revoked", sqlalchemy.Boolean, nullable=False)

    user = relationship("User", back_populates="refresh_tokens")
