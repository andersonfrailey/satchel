"""
This is the standings module used in pybaseball. I'm temporarily copying it
to deal with a bug in the package. Will be removed once the package is updated.
"""

# from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup
import requests


def standings(year: int) -> pd.DataFrame:
    url = f"https://www.baseball-reference.com/leagues/majors/{year}-standings.shtml"
    tables = pd.read_html(url)
    assert len(tables) == 6
    standings = pd.concat(tables)
    # remove text indicating playoff/division/wild card clinching
    standings["Tm"] = standings["Tm"].str.lstrip("y-").str.lstrip("x-").str.lstrip("w-")
    return standings


# def head_to_head(year: int) -> pd.DataFrame:
#     url = f"https://www.espn.com/mlb/standings/grid/_/year/{year}"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, "html.parser")
#     mod_content_div = soup.find("div", class_="mod-content")
#     if mod_content_div:
#         grid = pd.read_html(mod_content_div)
#     else:
#         raise ValueError()
#     return grid[0]  # TODO: Doublecheck the index here
