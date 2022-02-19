import pickle
from pathlib import Path
from satchel.model import Satchel

MSG = (
    "Model results differ from saved. If this is expected, run"
    "`createpickles.py` to create new expected new results"
)


def test_model(curpath):
    mod = Satchel(seed=123)
    res = mod.simulate(100)
    expectedres = pickle.load(Path(curpath, "basesim.p").open("rb"))

    assert res == expectedres, MSG


def test_transaction(transaction, curpath):
    mod = Satchel(seed=123, transactions=transaction)
    res = mod.simulate(100)
    expectedres = pickle.load(Path(curpath, "transactionsim.p").open("rb"))
    assert res == expectedres, MSG
