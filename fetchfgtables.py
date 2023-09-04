import time
import pandas as pd
from pathlib import Path
from satchel.utils import fetch_fg_projection_data
from satchel.constants import FG_PROJECTIONS
from datetime import datetime

CURPATH = Path(__file__).parent.resolve()
OUTPATH = Path(CURPATH, "fgprojections")


def main():
    # fetch depth chart projections
    for projection in FG_PROJECTIONS:
        print(projection)
        pit = fetch_fg_projection_data("pit", projection, datetime.today())
        pit["date"] = datetime.today()
        time.sleep(5)
        bat = fetch_fg_projection_data("bat", projection, datetime.today())
        bat["date"] = datetime.today()
        if projection == "steamer":
            append_results(Path(OUTPATH, f"{projection}_pit1.csv"), pit)
            append_results(Path(OUTPATH, f"{projection}_bat1.csv"), bat)
        elif projection == "zips":
            append_results(Path(OUTPATH, f"{projection}_pit1.csv"), pit)
            append_results(Path(OUTPATH, f"{projection}_bat1.csv"), bat)
        else:
            append_results(Path(OUTPATH, f"{projection}_pit.csv"), pit)
            append_results(Path(OUTPATH, f"{projection}_bat.csv"), bat)
        time.sleep(5)


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
