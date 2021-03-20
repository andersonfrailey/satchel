"""
Script to pull 40-man rosters from FanGraphs.
"""

import re
import time
import pickle
import pandas as pd
from requests_html import HTMLSession

BASE_URL = "https://www.fangraphs.com/roster-resource/depth-charts/{}"
TEAMS = [
    "diamondbacks",
    "braves",
    "orioles",
    "red-sox",
    "cubs",
    "white-sox",
    "reds",
    "cleveland",
    "rockies",
    "tigers",
    "marlins",
    "astros",
    "royals",
    "dodgers",
    "brewers",
    "mets",
    "phillies",
    "pirates",
    "padres",
    "giants",
    "cardinals",
    "nationals",
    "angels",
    "twins",
    "yankees",
    "athletics",
    "mariners",
    "rays",
    "rangers",
    "blue-jays",
]

session = HTMLSession()
all_ids = "name,pid,team\n"
id_data = {}
for team in TEAMS:
    teamids = []
    print(team)
    r = session.get(BASE_URL.format(team))
    r.html.render()
    time.sleep(1)
    tds = r.html.find(".cell-painted.roster-40")
    print(len(tds))
    for elm in tds:
        name = elm.full_text
        link = list(elm.absolute_links)[0]
        # player ID will be either a set of numbers for established MLBers, or
        # sa{dddd} for minor leage players. This regex should catch both
        m = re.search(r"\w+?\d+", link)
        pid = m.group()
        teamids.append(f"{name},{pid},{team}\n")
    teamset = set(teamids)
    print(len(teamset))
    all_ids += "".join(teamset)
    time.sleep(10)
with open("allids.csv", "w") as f:
    f.write(all_ids)
