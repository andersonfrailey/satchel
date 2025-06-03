"""
Combine all of the indivdual team schedules into a final file

NOTE: only put the CSV versions of each team's home schedule in their year
folder
"""

import time
import pandas as pd
import pytz
from pathlib import Path
from datetime import datetime

CUR_PATH = Path(__file__).resolve().parent
# update year, opening day, and final day when updating the file for a new season
YEAR = 2025
OPENING_DAY = "0327"
START_DATE = "0318"
FINAL_DAY = "0928"
SCHEDULE = Path(CUR_PATH, str(YEAR))

BASE_URL = (
    "https://www.ticketing-client.com/ticketing-client/csv/"
    "GameTicketPromotionPrice.tiksrv?team_id={team}&"
    "display_in=singlegame&ticket_category=Tickets&site_section=Default&"
    "sub_category=Default&leave_empty_games=true&event_type=T&year="
    "{year}&begin_date={year}{start_date}&end_date={year}{end_date}"
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
    "Athletics": "ATH",
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
    133: "ATH",
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

# Timezones
TZ_MAP = {
    "TOR": "-0400",
    "BAL": "-0400",
    "TBR": "-0400",
    "BOS": "-0400",
    "NYY": "-0400",
    "CLE": "-0400",
    "KCR": "-0500",
    "DET": "-0400",
    "MIN": "-0500",
    "CHW": "-0500",
    "LAA": "-0700",
    "HOU": "-0500",
    "ATH": "-0700",
    "SEA": "-0700",
    "TEX": "-0500",
    "ATL": "-0400",
    "MIA": "-0400",
    "NYM": "-0400",
    "WSN": "-0400",
    "PHI": "-0400",
    "MIL": "-0500",
    "STL": "-0500",
    "CHC": "-0500",
    "PIT": "-0400",
    "CIN": "-0400",
    "ARI": "-0600",
    "LAD": "-0700",
    "SFG": "-0700",
    "SPD": "-0700",
    "COL": "-0600",
}

# Games happening internationally
INTERNATIONAL_REGULAR_SEASON = {
    datetime(year=2025, month=3, day=18): "Tokyo Dome - Tokyo",
    datetime(year=2025, month=3, day=19): "Tokyo Dome - Tokyo",
}


def create_schedule(
    year: int = YEAR,
    start_date: str = OPENING_DAY,
    end_date: str = FINAL_DAY,
    outfile: Path | str | None = "",
    _return: bool = True,
    verbose: bool = False,
):
    def process(_id, team, year, start_date, verbose=False):
        if verbose:
            print(team)
        sched = pd.read_csv(
            BASE_URL.format(
                year=year, start_date=start_date, end_date=end_date, team=_id
            ),
            usecols=["START DATE", "SUBJECT", "START TIME", "LOCATION"],
            parse_dates=["START DATE"],
            date_format="%m/%d/%y",
        )
        if sched.empty:
            return sched
        sched["SUBJECT"] = sched["SUBJECT"].str.replace(" - Time TBD", "")
        sched[["away_team", "home_team"]] = sched["SUBJECT"].str.split(
            " at ", n=1, expand=True
        )
        sched["away"] = sched["away_team"].map(NAME_MAP)
        sched["home"] = sched["home_team"].map(NAME_MAP)
        # only keep games after opening day that are part of the regular season
        # games that get played internationally
        opening_day_dt = datetime.strptime(f"{OPENING_DAY}{YEAR}", "%m%d%Y")
        post_opener = sched["START DATE"] >= opening_day_dt
        international_openers = sched["START DATE"].isin(
            INTERNATIONAL_REGULAR_SEASON.keys()
        ) & sched["LOCATION"].isin(INTERNATIONAL_REGULAR_SEASON.values())
        keep_flag = post_opener | international_openers

        # convert date and start time to same timezone
        sched["START TIME"] = sched["START TIME"].fillna(
            "12:00 AM"
        )  # assign all missing times to midnight
        # TODO: Revisit this. Unclear if it should be mapped to home team or if time
        # is always local
        sched["tz"] = TZ_MAP[team]  # timezone
        sched["datetime"] = pd.to_datetime(
            sched["START DATE"].dt.strftime("%Y-%m-%d")
            + " "
            + sched["START TIME"]
            + " "
            + sched["tz"],
            format="%Y-%m-%d %I:%M %p %z",
            utc=True,
        )
        time.sleep(5)
        return sched[
            ["START DATE", "away", "home", "SUBJECT", "START TIME", "datetime"]
        ][keep_flag == True]

    dfs = [
        process(_id, team, year, start_date, verbose) for _id, team in ID_MAP.items()
    ]
    # TODO: Revist. When I figure out the timezone business can drop on datetime
    final_sched = pd.concat(dfs).drop_duplicates(subset=["START DATE", "SUBJECT"])
    if outfile:
        final_sched.to_csv(outfile, index=False)
    if _return:
        return final_sched


if __name__ == "__main__":
    create_schedule(
        year=YEAR,
        start_date=START_DATE,
        outfile=Path(CUR_PATH, f"schedule{YEAR}.csv"),
        _return=False,
        verbose=True,
    )
