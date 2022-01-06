import pickle
from pathlib import Path
from satchel.model import Satchel


def test_model(curpath):
    mod = Satchel(seed=123)
    res = mod.simulate(100)
    expectedres = pickle.load(Path(curpath, "basesim.p").open("rb"))
    import pdb

    pdb.set_trace()
    assert res == expectedres


# def test_transaction(transaction, curpath):
#     mod = Satchel(seed=123, transactions=transaction)
#     res = mod.simulate(100)
#     expectedres = pickle.load(Path(curpath, "transactionsim.p").open("rb"))
#     assert res == expectedres