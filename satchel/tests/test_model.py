from satchel.model import Satchel


def test_model():
    mod = Satchel(seed=123)
    res = mod.simulate(100)