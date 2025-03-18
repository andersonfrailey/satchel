"""
Script for pulling projections from FanGraphs and running Satchel
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from satchel.model import Satchel
from satchel.schedules.createschedule import YEAR
from satchel.constants import TEAM_ABBRS
from datetime import datetime
from collections import defaultdict


CURPATH = Path(__file__).parent.resolve()
OUTPATH = Path(CURPATH, "projections")


def main(percentiles: bool = False):

    # run Satchel
    print("Running Satchel")
    mod = Satchel(seed=856, cache=False, use_current_results=False)
    res = mod.simulate(20000)
    satchel_res = res.season_summary
    season_to_date = res.season_to_date()
    satchel_res["date"] = datetime.today()
    out = season_to_date.merge(
        satchel_res[
            [
                "Team",
                "Make Wild Card (%)",
                "Win Division (%)",
                "Win League (%)",
                "Win WS (%)",
            ]
        ],
        on="Team",
    )
    # out["date"] = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    print("Saving Satchel results")
    append_results(f"satchel{YEAR}.csv", out)
    if percentiles:
        percentiles = defaultdict(dict)
        for team in TEAM_ABBRS:
            for i in range(1, 163):
                percentiles[team][i] = res.season_percentile(team=team, wins=i)

        json.dump(percentiles, Path(OUTPATH, f"percentiles{YEAR}.json"), indent=4)


def append_results(outfile, results):
    """
    Read in the previous results and append the new results

    Parameters
    ----------
    outfile : str
        File the results will be appeneded to
    results : pd.DataFrame
        DataFrame containing the new projections
    """
    if Path(OUTPATH, outfile).exists():
        prev_res = pd.read_csv(Path(OUTPATH, outfile))
        all_res = pd.concat([prev_res, results])
    else:
        all_res = results
    all_res.to_csv(Path(OUTPATH, outfile), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--percentiles",
        action="store_true",
        help="If included, write out season percentiles",
    )
    args = parser.parse_args()
    main(args.percentiles)
