"""
Script for pulling projections from FanGraphs and running Satchel
"""

import pandas as pd
from pathlib import Path
from satchel.model import Satchel
from datetime import datetime


CURPATH = Path(__file__).parent.resolve()
OUTPATH = Path(CURPATH, "2024projections")


def main():

    # run Satchel
    print("Running Satchel")
    mod = Satchel(seed=856, cache=False)
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
    out["date"] = datetime.today()
    print("Saving Satchel results")
    append_results("satchel.csv", out)


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
    main()
