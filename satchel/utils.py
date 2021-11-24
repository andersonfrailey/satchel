"""
Utility functions for Satchel
"""
import pandas as pd
from pybaseball import playerid_lookup


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
