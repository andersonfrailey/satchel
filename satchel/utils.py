"""
Utility functions for Satchel
"""
import pandas as pd
import numpy as np
from pybaseball import playerid_lookup
from typing import Union


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
