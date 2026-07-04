from datetime import datetime
from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )


class CustomerIN(BaseConfig):
    name: str
    email: str
    phone: str
    user_id: int  # Assuming this is the ID of the user who created the customer


class Customer(CustomerIN):
    id: int


class CustomerOut(CustomerIN):
    id: int
    name: str
    email: str
    phone: str
    user_id: int  # Assuming this is the ID of the user who created the customer


class role_choice(str, Enum):
    customer = "customer"
    employee = "employee"
    warehouse_manager = "warehouse_manager"
    admin = "admin"


class UserCreate(BaseConfig):
    username: str
    password: str
    useremail: str
    userphone: str
    role: role_choice


class UserResponse(BaseConfig):
    id: int
    username: str
    useremail: str
    userphone: str
    role: role_choice
    is_active: bool
    created_at: datetime
    customer_id: int | None = None  # Optional field to link to customer profile if role is customer


class UserLogin(BaseConfig):
    useremail: str
    password: str


class TokenResponse(BaseConfig):
    access_token: str
    refresh_token: str


class RefreshTokenRequest(BaseConfig):
    refresh_token: str


class category_choice(str, Enum):
    Electronics = "Electronics"
    Cloths = "Cloths"
    Food = "Food"
    bed_bath_table = "bed_bath_table"
    garden_tools = "garden_tools"
    consoles_games = "consoles_games"
    health_beauty = "health_beauty"
    cool_stuff = "cool_stuff"
    perfumery = "perfumery"
    computers_accessories = "computers_accessories"
    watches_gifts = "watches_gifts"
    furniture_decor = "furniture_decor"


class PriceSuggestionRequest(BaseModel):
    category: category_choice
    inventory_level: int


class PriceSuggestionResponse(BaseModel):
    suggested_price: float


class ProductIN(BaseConfig):
    name: str
    category: category_choice
    cost_price: float
    current_price: float
    inventory_level: int = Field(..., gt=0, description="Inventory level must be positive")


class Product(BaseConfig):
    id: int
    name: str
    sku: str
    category: category_choice
    cost_price: float
    current_price: float
    inventory_level: int


class OrderItemIn(BaseConfig):
    item_id: int
    quantity: int = Field(..., gt=0, description="Quantity must be at least 1")


class orderItemOut(BaseConfig):
    id: int
    order_id: int
    item_id: int
    quantity: int
    price_at_purchase: float


class status_choice(str, Enum):
    pending = "pending"
    shipped = "shipped"
    delivered = "delivered"


class OrderIN(BaseConfig):
    status: status_choice
    cart_items: list[OrderItemIn] = Field(..., min_length=1, description="List of items in the cart")


class Order(OrderIN):
    id: int
    created_at: datetime
    order_total: float
    status: status_choice
    customer_id: int  # Assuming this is the ID of the customer who placed the order
    cart_items: list[orderItemOut] = Field(..., validation_alias=AliasChoices("items", "cart_items"))
