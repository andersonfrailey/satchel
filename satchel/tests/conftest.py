import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def transaction():
    return {"10954": {"team": "HOU", "date": "2021-04-01"}}


@pytest.fixture(scope="session")
def curpath():
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def schedule2021(curpath):
    return Path(curpath, "..", "schedules", "schedule2021.csv")
