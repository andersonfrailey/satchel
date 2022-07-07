"""
Combine all of the indivdual team schedules into a final file

NOTE: only put the CSV versions of each team's home schedule in their year
folder
"""
import time
from tracemalloc import start
import pandas as pd
from datetime import datetime
from pathlib import Path

CUR_PATH = Path(__file__).resolve().parent
# update year and opening day when updating the file
YEAR = 2022
OPENING_DAY = "0331"
SCHEDULE = Path(CUR_PATH, str(YEAR))

BASE_URL = (
    "https://www.ticketing-client.com/ticketing-client/csv/"
    "GameTicketPromotionPrice.tiksrv?team_id={team}&home_team_id={team}&"
    "display_in=singlegame&ticket_category=Tickets&site_section=Default&"
    "sub_category=Default&leave_empty_games=true&event_type=T&year="
    "{year}&begin_date={year}{start_date}"
)

NAME_MAP = {
    "Diamondbacks": "ARI",
    "D-backs": "ARI",
    "Braves": "ATL",
    "Orioles": "BAL",
    "Red Sox": "BOS",
    "Cubs": "CHC",
    "White Sox": "CHW",
    "Reds": "CIN",
    "Guardians": "CLE",
    "Rockies": "COL",
    "Tigers": "DET",
    "Marlins": "MIA",
    "Astros": "HOU",
    "Royals": "KCR",
    "Dodgers": "LAD",
    "Brewers": "MIL",
    "Mets": "NYM",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Padres": "SDP",
    "Giants": "SFG",
    "Cardinals": "STL",
    "Nationals": "WSN",
    "Angels": "LAA",
    "Twins": "MIN",
    "Yankees": "NYY",
    "Athletics": "OAK",
    "Mariners": "SEA",
    "Rays": "TBR",
    "Rangers": "TEX",
    "Blue Jays": "TOR",
}

ID_MAP = {
    141: "TOR",
    110: "BAL",
    139: "TBR",
    111: "BOS",
    147: "NYY",
    114: "CLE",
    118: "KCR",
    116: "DET",
    142: "MIN",
    145: "CHW",
    108: "LAA",
    117: "HOU",
    133: "OAK",
    136: "SEA",
    140: "TEX",
    144: "ATL",
    146: "MIA",
    121: "NYM",
    120: "WSN",
    143: "PHI",
    158: "MIL",
    138: "STL",
    112: "CHC",
    134: "PIT",
    113: "CIN",
    109: "ARI",
    119: "LAD",
    137: "SFG",
    135: "SPD",
    115: "COL",
}


def create_schedule(
    year: int = YEAR,
    start_date: str = OPENING_DAY,
    outfile: str = "",
    _return: bool = True,
    verbose: bool = False,
):
    def process(_id, team, year, start_date, verbose=False):
        if verbose:
            print(team)
        sched = pd.read_csv(
            BASE_URL.format(year=year, start_date=start_date, team=_id),
            usecols=["START DATE", "SUBJECT"],
        )
        sched["START DATE"] = pd.to_datetime(sched["START DATE"])
        sched["SUBJECT"] = sched["SUBJECT"].str.replace(" - Time TBD", "")
        sched[["away_team", "home_team"]] = sched["SUBJECT"].str.split(
            " at ", 1, expand=True
        )
        sched["away"] = sched["away_team"].map(NAME_MAP)
        sched["home"] = sched["home_team"].map(NAME_MAP)
        time.sleep(1)
        return sched[["START DATE", "away", "home", "SUBJECT"]]

    dfs = [
        process(_id, team, year, start_date, verbose) for _id, team in ID_MAP.items()
    ]
    final_sched = pd.concat(dfs)
    if outfile:
        final_sched.to_csv(outfile, index=False)
    if _return:
        return final_sched


if __name__ == "__main__":
    create_schedule(
        year=YEAR,
        start_date=OPENING_DAY,
        outfile=Path(CUR_PATH, f"schedule{YEAR}.csv"),
        _return=False,
        verbose=True,
    )
