"""Pytest configuration and fixtures for backend tests."""
from unittest.mock import MagicMock
import pytest
from app.api.v1.deps import get_db
from app.main import app

@pytest.fixture(autouse=True)
def mock_db_if_no_postgres():
    """Mock the DB session for fast offline test execution."""
    mock_session = MagicMock()
    mock_session.execute.return_value.scalars.return_value.all.return_value = []
    mock_session.execute.return_value.scalars.return_value.first.return_value = None
    mock_session.execute.return_value.scalar_one_or_none.return_value = None
    
    def override_get_db():
        try:
            yield mock_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
