import pickle
from pathlib import Path

import pytest
from satchel.model import Satchel

MSG = (
    "Model results differ from saved. If this is expected, run"
    "`createpickles.py` to create new expected new results"
)


def test_model(curpath, schedule2021, batter_projections, pitcher_projections):
    mod = Satchel(
        seed=123,
        schedule=schedule2021,
        use_current_results=False,
        batter_proj=batter_projections,
        pitcher_proj=pitcher_projections,
    )
    res = mod.simulate(100)
    expectedres = pickle.load(Path(curpath, "basesim.p").open("rb"))

    assert res == expectedres, MSG
    assert mod.talent["final_talent"].isna().sum() == 0

    mod = Satchel(
        seed=123,
        schedule=schedule2021,
        use_current_results=False,
        batter_proj=batter_projections,
        pitcher_proj=pitcher_projections,
    )
    res = mod.simulate(100, probability_method="elo", elo_scale=400)

    with pytest.raises(ValueError):
        res = mod.simulate(100, probability_method="newton")


def test_transaction(
    transaction, curpath, schedule2021, batter_projections, pitcher_projections
):
    mod = Satchel(
        seed=123,
        transactions=transaction,
        schedule=schedule2021,
        use_current_results=False,
        batter_proj=batter_projections,
        pitcher_proj=pitcher_projections,
    )
    res = mod.simulate(100)
    expectedres = pickle.load(Path(curpath, "transactionsim.p").open("rb"))
    assert res == expectedres, MSG
    assert mod.talent["final_talent"].isna().sum() == 0


def test_warning(schedule2021):
    with pytest.warns(UserWarning):
        mod = Satchel(schedule=schedule2021, cache=False)
        assert mod.talent["final_talent"].isna().sum() == 0
