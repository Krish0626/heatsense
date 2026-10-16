"""Declarative base shared by every SQLAlchemy model.

Import every model module here so Alembic (once configured) can see them
via Base.metadata for autogeneration.
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
