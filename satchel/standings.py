"""
This is the standings module used in pybaseball. I'm temporarily copying it
to deal with a bug in the package. Will be removed once the package is updated.
"""

# from typing import List, Optional

import pandas as pd


def standings(year: int) -> pd.DataFrame:
    url = f"https://www.baseball-reference.com/leagues/majors/{year}-standings.shtml"
    tables = pd.read_html(url)
    assert len(tables) == 6
    standings = pd.concat(tables)
    # remove text indicating playoff/division/wild card clinching
    standings["Tm"] = standings["Tm"].str.lstrip("y-").str.lstrip("x-").str.lstrip("w-")
    return standings
