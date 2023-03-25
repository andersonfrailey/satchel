"""
Script for pulling projections from FanGraphs and running Satchel
"""
import pandas as pd
from pathlib import Path
from satchel.model import Satchel
from datetime import datetime


CURPATH = Path(__file__).parent.resolve()
OUTPATH = Path(CURPATH, "2023projections")


def main():
    # pull FanGraphs data
    print("Fetching FanGraphs projections")
    proj = pd.read_html(
        "https://www.fangraphs.com/depthcharts.aspx?position=Standings"
    )[7]
    fg_proj = pd.concat(
        [proj["Unnamed: 0_level_0"], proj["2023 Projected Full Season"]], axis=1
    )
    fg_proj["date"] = datetime.today()
    print("Saving FanGraphs results")
    append_results("fangraphs.csv", fg_proj)

    # run Satchel
    print("Running Satchel")
    mod = Satchel(seed=856)
    res = mod.simulate(20000)
    satchel_res = res.season_summary
    satchel_res["date"] = datetime.today()
    print("Saving Satchel results")
    append_results("satchel.csv", satchel_res)


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
