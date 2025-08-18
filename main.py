import os
from datetime import date, datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Date, DateTime, ForeignKey,
    create_engine, select, func, event
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

# -------------------- Config --------------------
DB_URL = os.getenv("DB_URL", "sqlite:///./shopping.db")
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
from sqlalchemy import event

def make_engine(url: str):
    if url.startswith("sqlite"):
        eng = create_engine(url, connect_args={"check_same_thread": False})
        @event.listens_for(eng, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
        return eng
    else:
        return create_engine(url, pool_pre_ping=True)

engine = make_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()

# Enforce FK in SQLite (extra safety; we also guard in code)
if DB_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

app = FastAPI(title="Shopping List Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- DB Models --------------------
class ProductGroup(Base):
    __tablename__ = "product_groups"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)

    products = relationship("Product", back_populates="group", cascade="all, delete")


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, ForeignKey("product_groups.id"), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text, default="")
    default_unit = Column(String(32), default="pcs")

    group = relationship("ProductGroup", back_populates="products")


class ShoppingList(Base):
    __tablename__ = "shopping_lists"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(64), index=True, nullable=False)
    name = Column(String(200), nullable=False)
    shopping_date = Column(Date, nullable=True)
    status = Column(String(16), default="active")  # active|completed
    created_at = Column(DateTime, default=datetime.utcnow)

    items = relationship("ShoppingListItem", back_populates="list", cascade="all, delete")


class ShoppingListItem(Base):
    __tablename__ = "shopping_list_items"
    id = Column(Integer, primary_key=True)
    list_id = Column(Integer, ForeignKey("shopping_lists.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    unit = Column(String(32), default="pcs")
    purchased = Column(Boolean, default=False)

    list = relationship("ShoppingList", back_populates="items")
    product = relationship("Product")

# -------------------- Pydantic Schemas --------------------
class ProductOut(BaseModel):
    id: int
    group_id: int
    name: str
    description: str
    default_unit: str
    class Config: from_attributes = True

class ProductGroupOut(BaseModel):
    id: int
    name: str
    class Config: from_attributes = True

class NewListItem(BaseModel):
    product_id: int
    quantity: int = Field(1, ge=1)
    unit: Optional[str] = None

class ShoppingListCreate(BaseModel):
    name: str
    shopping_date: Optional[date] = None
    items: List[NewListItem]

class ShoppingListItemOut(BaseModel):
    id: int
    product_id: int
    quantity: int
    unit: str
    purchased: bool
    product: ProductOut
    class Config: from_attributes = True

class ShoppingListOut(BaseModel):
    id: int
    name: str
    shopping_date: Optional[date]
    status: str
    created_at: datetime
    items: List[ShoppingListItemOut] = []
    class Config: from_attributes = True

class ItemPatch(BaseModel):
    quantity: Optional[int] = Field(None, ge=1)
    unit: Optional[str] = None
    purchased: Optional[bool] = None

# ----- Admin request schemas -----
class ProductGroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)

class ProductGroupPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)

class ProductCreate(BaseModel):
    group_id: int
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = ""
    default_unit: Optional[str] = "pcs"

class ProductPatch(BaseModel):
    group_id: Optional[int] = None
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    default_unit: Optional[str] = None

# -------------------- Dependencies --------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_id(x_user_id: Optional[str] = Header(default=None)):
    """
    For now, we take the user id from an HTTP header `x-user-id`.
    In Telegram WebApp, you can parse tg.initDataUnsafe.user.id on the frontend and pass it here.
    Later we can verify initData signature if needed.
    """
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing x-user-id header")
    return x_user_id

# -------------------- Catalog Endpoints --------------------
@app.get("/product-groups", response_model=List[ProductGroupOut])
def list_groups(db: Session = Depends(get_db)):
    return db.execute(select(ProductGroup).order_by(ProductGroup.id)).scalars().all()

@app.get("/products", response_model=List[ProductOut])
def list_products(
    group_id: Optional[int] = None,
    q: Optional[str] = None,
    db: Session = Depends(get_db)
):
    stmt = select(Product)
    if group_id:
        stmt = stmt.where(Product.group_id == group_id)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(
            func.lower(Product.name).like(func.lower(like)) |
            func.lower(Product.description).like(func.lower(like))
        )
    return db.execute(stmt.order_by(Product.id.desc())).scalars().all()

# -------------------- Admin: Product Groups --------------------
@app.post("/product-groups", response_model=ProductGroupOut, status_code=201)
def create_group(payload: ProductGroupCreate, db: Session = Depends(get_db)):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    g = ProductGroup(name=name)
    db.add(g)
    db.commit()
    db.refresh(g)
    return g

@app.patch("/product-groups/{group_id}", response_model=ProductGroupOut)
def patch_group(group_id: int, payload: ProductGroupPatch, db: Session = Depends(get_db)):
    g = db.get(ProductGroup, group_id)
    if not g:
        raise HTTPException(status_code=404, detail="Group not found")
    if payload.name is not None:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        g.name = name
    db.commit()
    db.refresh(g)
    return g

@app.delete("/product-groups/{group_id}", status_code=204)
def delete_group(group_id: int, db: Session = Depends(get_db)):
    g = db.get(ProductGroup, group_id)
    if not g:
        raise HTTPException(status_code=404, detail="Group not found")

    product_ids = [p.id for p in g.products]
    if product_ids:
        used = db.execute(
            select(ShoppingListItem).where(ShoppingListItem.product_id.in_(product_ids))
        ).first()
        if used:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete group: some products are used in shopping lists"
            )

    db.delete(g)
    db.commit()
    return

# -------------------- Admin: Products --------------------
@app.post("/products", response_model=ProductOut, status_code=201)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    group = db.get(ProductGroup, payload.group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    p = Product(
        group_id=payload.group_id,
        name=payload.name.strip(),
        description=(payload.description or "").strip(),
        default_unit=(payload.default_unit or "pcs").strip(),
    )
    if not p.name:
        raise HTTPException(status_code=400, detail="Name is required")
    db.add(p)
    db.commit()
    db.refresh(p)
    return p

@app.patch("/products/{product_id}", response_model=ProductOut)
def patch_product(product_id: int, payload: ProductPatch, db: Session = Depends(get_db)):
    p = db.get(Product, product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")

    if payload.group_id is not None:
        group = db.get(ProductGroup, payload.group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Target group not found")
        p.group_id = payload.group_id

    if payload.name is not None:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        p.name = name

    if payload.description is not None:
        p.description = (payload.description or "").strip()

    if payload.default_unit is not None:
        unit = (payload.default_unit or "").strip() or "pcs"
        p.default_unit = unit

    db.commit()
    db.refresh(p)
    return p

@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    p = db.get(Product, product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")

    used = db.execute(
        select(ShoppingListItem).where(ShoppingListItem.product_id == product_id)
    ).first()
    if used:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete product: it is used in shopping lists"
        )

    db.delete(p)
    db.commit()
    return

# -------------------- Shopping List Endpoints --------------------
@app.post("/shopping-lists", response_model=ShoppingListOut, status_code=201)
def create_list(payload: ShoppingListCreate, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)):
    if not payload.items:
        raise HTTPException(status_code=400, detail="List must contain at least one item")

    sl = ShoppingList(
        user_id=user_id,
        name=payload.name.strip(),
        shopping_date=payload.shopping_date,
        status="active",
    )
    db.add(sl)
    db.flush()

    for it in payload.items:
        product = db.get(Product, it.product_id)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {it.product_id} not found")

        unit = it.unit or product.default_unit
        db.add(ShoppingListItem(
            list_id=sl.id,
            product_id=product.id,
            quantity=it.quantity,
            unit=unit,
            purchased=False
        ))

    db.commit()
    db.refresh(sl)
    _ = sl.items
    return sl

@app.get("/shopping-lists", response_model=List[ShoppingListOut])
def get_lists(status: Optional[str] = None, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)):
    stmt = select(ShoppingList).where(ShoppingList.user_id == user_id)
    if status in {"active", "completed"}:
        stmt = stmt.where(ShoppingList.status == status)
    return db.execute(stmt.order_by(ShoppingList.created_at.desc())).scalars().all()

@app.get("/shopping-lists/{list_id}", response_model=ShoppingListOut)
def get_list(list_id: int, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)):
    sl = db.get(ShoppingList, list_id)
    if not sl or sl.user_id != user_id:
        raise HTTPException(status_code=404, detail="List not found")
    _ = sl.items
    return sl

@app.patch("/shopping-lists/{list_id}/complete", response_model=ShoppingListOut)
def complete_list(list_id: int, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)):
    sl = db.get(ShoppingList, list_id)
    if not sl or sl.user_id != user_id:
        raise HTTPException(status_code=404, detail="List not found")
    sl.status = "completed"
    db.commit()
    db.refresh(sl)
    _ = sl.items
    return sl

# -------------------- Item Endpoints --------------------
@app.patch("/shopping-list-items/{item_id}", response_model=ShoppingListItemOut)
def patch_item(item_id: int, payload: ItemPatch, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)):
    item = db.get(ShoppingListItem, item_id)
    if not item or item.list.user_id != user_id:
        raise HTTPException(status_code=404, detail="Item not found")

    if payload.quantity is not None:
        item.quantity = payload.quantity
    if payload.unit is not None:
        item.unit = payload.unit
    if payload.purchased is not None:
        item.purchased = payload.purchased

    db.commit()
    db.refresh(item)
    return item

@app.delete("/shopping-list-items/{item_id}", status_code=204)
def delete_item(item_id: int, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)):
    item = db.get(ShoppingListItem, item_id)
    if not item or item.list.user_id != user_id:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(item)
    db.commit()
    return

# -------------------- Bootstrap --------------------
def seed_if_empty(db: Session):
    # Seed groups/products if empty
    if db.execute(select(ProductGroup)).first():
        return

    fruits = ProductGroup(name="Fruits")
    dairy = ProductGroup(name="Dairy")
    bakery = ProductGroup(name="Bakery")
    db.add_all([fruits, dairy, bakery])
    db.flush()

    products = [
        Product(group_id=fruits.id, name="Apple", description="Fresh red apples", default_unit="pcs"),
        Product(group_id=fruits.id, name="Banana", description="Ripe bananas", default_unit="pcs"),
        Product(group_id=dairy.id,  name="Milk", description="1L whole milk", default_unit="L"),
        Product(group_id=dairy.id,  name="Cheese", description="Cheddar block", default_unit="g"),
        Product(group_id=bakery.id, name="Bread", description="Whole grain loaf", default_unit="loaf"),
    ]
    db.add_all(products)
    db.commit()

Base.metadata.create_all(engine)
with SessionLocal() as _db:
    seed_if_empty(_db)

# Run local:
# uvicorn main:app --reload
