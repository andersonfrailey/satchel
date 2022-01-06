import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def transaction():
    return {"10954": "HOU"}


@pytest.fixture(scope="session")
def curpath():
    return Path(__file__).resolve().parent
