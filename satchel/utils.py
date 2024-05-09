"""
Utility functions for Satchel
"""

import requests
import json
import pandas as pd
import numpy as np
from pybaseball import playerid_lookup
from typing import Union
from functools import lru_cache
from .constants import FG_PROJECTIONS


FG_API = "https://www.fangraphs.com/api/projections?stats={stats}&type={proj}"


def player_id_lookup(last=None, first=None, fuzzy=False):
    """Find a player's FanGraphs ID

    Parameters
    ----------
    last : str, optional
        Player last name, by default None
    first : str, optional
        Player first name, by default None
    fuzzy : bool, optional
        If an exact match isn't found, return 5 closest names, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with the player names and FanGraphs IDs that match the search
    """
    res = playerid_lookup(last, first, fuzzy)
    return res[["name_last", "name_first", "key_fangraphs"]].copy()


def probability_calculations(
    team1_talent: Union[float, int, np.array],
    team2_talent: Union[float, int, np.array],
    probability_method: str = "bradley_terry",
    elo_scale: int = 400,
):
    if probability_method == "bradley_terry":
        return np.exp(team1_talent) / (np.exp(team1_talent) + np.exp(team2_talent))
    elif probability_method == "elo":
        exp_val = (team1_talent - team2_talent) / elo_scale
        return 1 / (1 + np.power(10, exp_val))
    else:
        raise ValueError("`probability_method must be `bradley_terry` or `elo`.")


@lru_cache(maxsize=5)
def fetch_fg_projection_data(stats: str, fg_projection: str, date):
    """
    Fetch projection data from FanGraphs

    Parameters
    ----------
    stats : str
        `pit` if fetching for pitcher projections. `bat` if fetching
        batter projections
    fg_projection : str
        Which FG projections to fetch
    date
        Date the data was fetched on
    """
    if fg_projection not in FG_PROJECTIONS:
        raise ValueError(f"`fg_projections` must be in {FG_PROJECTIONS}")
    req = requests.get(FG_API.format(stats=stats, proj=fg_projection))
    if req.status_code != 200:
        raise ConnectionError("Connection to FanGraphs failed")
    data = pd.DataFrame(json.loads(req.content))
    if "playerid" not in data.columns:
        data.rename(columns={"playerids": "playerid"}, inplace=True)
    return data
