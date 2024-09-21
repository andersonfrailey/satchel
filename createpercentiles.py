import pandas as pd
import pickle
import json
from satchel.constants import TEAM_ABBRS
from collections import defaultdict


def create_percentiles():
    # Load the data
    with open("2024projections/openingday2024.p", "rb") as f:
        results = pickle.load(f)
    percentiles = {}
    for team in TEAM_ABBRS:
        _percentiles = {}
        for i in range(1, 163):
            _percentiles[i] = results.season_percentile(team, i)
        percentiles[team] = _percentiles

    # save percentiles as a json
    with open("2024projections/percentiles.json", "w") as f:
        json.dump(percentiles, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    create_percentiles()
