"""
Create the initial pickle files for testing
"""
import pickle
from satchel.model import Satchel
from pathlib import Path

s1 = Satchel(seed=123)
r1 = s1.simulate(100)
# print(r1.ws_counter)
# print(s1.teams)
# print(r1.noise)

s2 = Satchel(seed=123, transactions={"10954": {"team": "HOU", "date": "2021-04-01"}})
r2 = s2.simulate(100)
# print(r2.ws_counter)
# print(s2.teams)
# print(r1 == r2)
pickle.dump(r1, Path("satchel/tests/basesim.p").open("wb"))
pickle.dump(r2, Path("satchel/tests/transactionsim.p").open("wb"))
