from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def transaction():
    return {"10954": {"team": "HOU", "date": "2025-04-01"}}


@pytest.fixture(scope="session")
def curpath():
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def schedule2025(curpath):
    return Path(curpath, "..", "schedules", "schedule2025.csv")


@pytest.fixture(scope="session")
def batter_projections(curpath):
    return Path(curpath, "..", "data", "batterprojections_test.csv")


@pytest.fixture(scope="session")
def pitcher_projections(curpath):
    return Path(curpath, "..", "data", "pitcherprojections_test.csv")
