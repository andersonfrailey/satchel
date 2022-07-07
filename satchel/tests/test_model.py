import pickle
from pathlib import Path
from satchel.model import Satchel

MSG = (
    "Model results differ from saved. If this is expected, run"
    "`createpickles.py` to create new expected new results"
)


def test_model(curpath, schedule2021):
    mod = Satchel(seed=123, schedule=schedule2021, use_current_standings=False)
    res = mod.simulate(100)
    expectedres = pickle.load(Path(curpath, "basesim.p").open("rb"))

    assert res == expectedres, MSG


def test_transaction(transaction, curpath, schedule2021):
    mod = Satchel(
        seed=123,
        transactions=transaction,
        schedule=schedule2021,
        use_current_standings=False,
    )
    res = mod.simulate(100)
    expectedres = pickle.load(Path(curpath, "transactionsim.p").open("rb"))
    assert res == expectedres, MSG
