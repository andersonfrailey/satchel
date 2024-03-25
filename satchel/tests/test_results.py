import numpy as np
from pathlib import Path

import pytest
from satchel.model import Satchel


def test_results():
    mod = Satchel(seed=123)
    res = mod.simulate(100)
    # test that all seasons end in 162 games
    assert np.allclose(res.results_df[["wins", "losses"]].sum(axis=1), 162)
