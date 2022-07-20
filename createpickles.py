"""
Create the initial pickle files for testing
"""
import pickle
from satchel.model import Satchel
from pathlib import Path

schedule2021 = Path("satchel", "schedules", "schedule2021.csv")
batter_projections = Path("satchel", "data", "batterprojections_test.csv")
pitcher_projections = Path("satchel", "data", "pitcherprojections_test.csv")

s1 = Satchel(
    seed=123,
    schedule=schedule2021,
    batter_proj=batter_projections,
    pitcher_proj=pitcher_projections,
    use_current_results=False,
    cache=False,
)
r1 = s1.simulate(100)

s2 = Satchel(
    seed=123,
    transactions={"10954": {"team": "HOU", "date": "2021-04-01"}},
    schedule=schedule2021,
    batter_proj=batter_projections,
    pitcher_proj=pitcher_projections,
    use_current_results=False,
    cache=False,
)
r2 = s2.simulate(100)
pickle.dump(r1, Path("satchel", "tests", "basesim.p").open("wb"))
pickle.dump(r2, Path("satchel", "tests", "transactionsim.p").open("wb"))
