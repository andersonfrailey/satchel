"""
This moduel contains the heart of satchel. All of the season will be simulated
from this main class
"""
import pandas as pd
import numpy as np
import difflib
from . import constants
from .modelresults import SatchelResults
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Callable
from tqdm import tqdm


CUR_PATH = Path(__file__).resolve().parent
DATA_PATH = Path(CUR_PATH, "data")
SCHEDUEL_PATH = Path(CUR_PATH, "schedules", "schedule2021.csv")


class Satchel:
    def __init__(
        self,
        talent_measure: str = "median",
        transactions: dict = None,
        noise: bool = True,
        seed: int = None,
    ):
        """Main model class

        Parameters
        ----------
        n : int, optional
            number of times to run the simulation, by default 10000
        playoffs : function, optional
            function used to simulate the playoffs, by default standard_playoff
        seed : int, float, optional
            seed used for random draws, by default None
        """
        if talent_measure.lower() not in ["median", "mean"]:
            raise ValueError("`talent_measure` must be median or mean")
        self.talent_measure = talent_measure
        self.transactions = transactions
        self.talent = self._calculate_talent(transactions)
        self.schedule = pd.read_csv(SCHEDUEL_PATH)
        self.teams = constants.DIVS.keys()
        self.random = np.random.default_rng(seed)
        self.noise = noise

    def simulate(
        self,
        n: int = 10000,
        noise: bool = True,
    ) -> SatchelResults:
        """Run a model simulation n times

        Parameters
        ----------
        n : int, optional
            Number of iterations to run the model for, by default 10000
        noise : bool, optional
            Whether or not to add noise to team talent levels, by default True

        Returns
        -------
        SatchelResults
            Instance of the SatchelResults class.
        """
        # merge schedule and WAR data to get our dataset for simulations
        data = pd.merge(
            self.schedule,
            self.talent[["Team", "talent"]],
            left_on="away",
            right_on="Team",
        )
        data = pd.merge(
            data, self.talent[["Team", "talent"]], left_on="home", right_on="Team"
        )
        # clean up data after merge
        data.drop(["Team_x", "Team_y"], axis=1, inplace=True)
        data.rename(
            columns={"talent_x": "away_talent", "talent_y": "home_talent"}, inplace=True
        )
        # counters to track outcomes
        ws_counter = Counter()  # world series championships
        league_counter = Counter()  # league championships
        div_counter = Counter()  # division championships
        wc_counter = Counter()  # wild card appearances
        playoff_counter = Counter()  # any postseason appearance
        all_results = []
        all_matchups = []
        all_noise = []  # the talent noise for a given team in a season
        full_seasons = []  # hold all of the results for each season
        for i in tqdm(range(n)):
            (
                results,
                playoffs,
                div_winners,
                wc_winners,
                matchups,
                noise,
                full_season,
            ) = self.simseason(
                data,
            )
            ws_counter.update([playoffs["ws"]])
            div_counter.update(div_winners["Team"])
            league_counter.update([playoffs["nl"]["cs"]])
            league_counter.update([playoffs["al"]["cs"]])
            wc_counter.update(wc_winners["Team"])
            playoff_counter.update(wc_winners["Team"])
            playoff_counter.update(div_winners["Team"])
            results["sim"] = i
            all_results.append(results)
            all_matchups.append(matchups)
            all_noise.append(noise)
            full_seasons.append(full_season)

        return SatchelResults(
            ws_counter,
            league_counter,
            div_counter,
            wc_counter,
            playoff_counter,
            pd.concat(all_results),
            pd.DataFrame(all_matchups),
            self.talent,
            n,
            self.transactions,
            self.schedule,
            data,
            noise,
            full_seasons,
        )

    def simseason(self, data) -> tuple:
        """Run full simulation of a single season

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing merged talent and schedule information.

        Returns
        -------
        tuple
            Tuple containing final results, wild card and division winners, and
            playoff matchups.
        """
        data["h_talent"] = data["home_talent"]
        data["a_talent"] = data["away_talent"]
        _talent = self.talent[["Team", "talent"]].set_index("Team").to_dict("index")
        # add random noise to team talent for the season
        if self.noise:
            _noise = self.random.normal(
                scale=self.talent["talent"].std(), size=len(self.teams)
            )
            team_noise = {team: _noise[i] for i, team in enumerate(self.teams)}
            data["a_talent"] += np.array([team_noise[team] for team in data["away"]])
            data["h_talent"] += np.array([team_noise[team] for team in data["home"]])
            for team, value in team_noise.items():
                _talent[team]["talent"] += value
        # sim regular season
        home_win_prob = np.exp(data["h_talent"]) / (
            np.exp(data["a_talent"]) + np.exp(data["h_talent"])
        )
        probs = self.random.random(len(data))
        winner = pd.Series(
            np.where(home_win_prob >= probs, data["home"], data["away"]), name="wins"
        )
        loser = pd.Series(
            np.where(home_win_prob >= probs, data["away"], data["home"]), name="losses"
        )
        data["winner"] = winner
        data["loser"] = loser
        results = pd.concat(
            [winner.value_counts(), loser.value_counts()], axis=1
        ).reset_index()
        results.rename(columns={"index": "Team"}, inplace=True)
        results["league"] = results["Team"].map(constants.LEAGUE)
        results["division"] = results["Team"].map(constants.DIV)

        # post season play
        (
            final_res,
            cs_winners,
            div_winners,
            wc_winners,
            matchups,
        ) = self.standard_playoff(results, _talent)
        # column for season result
        results["season_result"] = np.where(
            results["Team"] == final_res["ws"],
            "Win World Series",
            np.where(
                results["Team"].isin(cs_winners),
                "Win League",
                np.where(
                    results["Team"].isin(div_winners["Team"]),
                    "Division Champ",
                    np.where(
                        results["Team"].isin(wc_winners["Team"]),
                        "Wild Card",
                        "Missed Playoff",
                    ),
                ),
            ),
        )
        return results, final_res, div_winners, wc_winners, matchups, team_noise, data

    def standard_playoff(self, results, talent, n_divwinners=1, n_wildcard=2):
        def sim_round(teams, talent, n_games):
            """
            Simulate a playoff round with n_games
            """
            team1 = 0
            team2 = 0
            team1_win_prob = np.exp(talent[teams[0]]["talent"]) / (
                np.exp(talent[teams[0]]["talent"]) + np.exp(talent[teams[1]]["talent"])
            )
            for _ in range(n_games):
                prob = self.random.random()
                if team1_win_prob >= prob:
                    team1 += 1
                    continue
                team2 += 1
            if team1 > team2:
                return teams[0]
            return teams[1]

        def league_sim(wc_winners, div_winners, talent, league, matchups):
            """
            Simulate a leagues half of the postseason
            """
            # sort and join the teams so that all match ups count the same
            wc = "-".join(
                sorted([wc_winners["Team"].iloc[0], wc_winners["Team"].iloc[1]])
            )
            matchups[f"{league} Wild Card"] = wc
            wc_winner = sim_round(
                [wc_winners["Team"].iloc[0], wc_winners["Team"].iloc[1]], talent, 1
            )
            matchups[f"{league} WC Champ"] = wc_winner
            ds1 = "-".join(sorted([wc_winner, div_winners["Team"].iloc[0]]))
            matchups[f"{league}DS 1"] = ds1
            div_rd1 = sim_round([wc_winner, div_winners["Team"].iloc[0]], talent, 5)
            matchups[f"{league}DS 1 Champ"] = div_rd1
            ds2 = "-".join(
                sorted([div_winners["Team"].iloc[1], div_winners["Team"].iloc[2]])
            )
            matchups[f"{league}DS 2"] = ds2
            div_rd2 = sim_round(
                [div_winners["Team"].iloc[1], div_winners["Team"].iloc[2]], talent, 5
            )
            matchups[f"{league}DS 2 Champ"] = div_rd2
            matchups[f"{league}CS"] = "-".join(sorted([div_rd1, div_rd2]))
            cs = sim_round([div_rd1, div_rd2], talent, 7)
            matchups[f"{league} Champ"] = cs
            return (
                {"wc": wc_winner, "div_rd1": div_rd1, "div_rd2": div_rd2, "cs": cs},
                matchups,
            )

        div_winners = results.groupby(["league", "division"]).apply(
            pd.DataFrame.nlargest, n=n_divwinners, columns="wins"
        )
        # take out division winners and pull the top remaining teams 4 wild card
        wc = results[~results["Team"].isin(div_winners["Team"])]
        wc_winners = wc.groupby("league").apply(
            pd.DataFrame.nlargest, n=n_wildcard, columns="wins"
        )
        # simulate all the rounds
        nlres, matchups = league_sim(
            wc_winners.loc["NL"], div_winners.loc["NL"], talent, "NL", {}
        )
        alres, matchups = league_sim(
            wc_winners.loc["AL"], div_winners.loc["AL"], talent, "AL", matchups
        )
        # world series winner
        matchups["World Series"] = "-".join(sorted([nlres["cs"], alres["cs"]]))
        champ = sim_round([nlres["cs"], alres["cs"]], talent, 7)
        matchups["WS Winner"] = champ
        return (
            {"nl": nlres, "al": alres, "ws": champ},
            [alres["cs"], nlres["cs"]],
            div_winners,
            wc_winners,
            matchups,
        )

    def matchup(self, team1: str, team2: str) -> float:
        """Calculate the probability of two teams winning when they play each other

        Parameters
        ----------
        team1 : str
            First team in the matchup
        team2 : str
            Second team in the matchup

        Returns
        -------
        float
            Tuple with each team's win probability: (team1, team2)

        Raises
        ------
        ValueError
            Raised if either team1 or team2 is an invalid team name
        """
        # assert that the two teams are actual team abbreviations
        if team1.upper() not in self.teams:
            similar = difflib.get_close_matches(team1, self.teams)
            msg = f"{team1} is not valid. Similar teams are: {similar}"
            raise ValueError(msg)
        if team2.upper() not in self.teams:
            similar = difflib.get_close_matches(team2, self.teams)
            msg = f"{team2} is not valid. Similar teams are: {similar}"
            raise ValueError(msg)
        team1_talent = self.talent["talent"][self.talent["Team"] == team1].values
        team2_talent = self.talent["talent"][self.talent["Team"] == team2].values
        team1_prob = np.exp(team1_talent) / (
            np.exp(team1_talent) + np.exp(team2_talent)
        )

        return team1_prob[0], 1 - team1_prob[0]

    ####### Private methods #######

    def _calculate_talent(self, transactions=None):
        """Private method used to calculate each team's talent level by taking
        the average of ZiPS and Steamer WAR projections on FanGraphs

        Parameters
        ----------
        transactions : dict, optional
            Dictionary containing transaction information.

        Returns
        -------
        pd.DataFrame
            DataFrame containing talent levels for each team.
        """
        active = pd.read_csv(Path(DATA_PATH, "activeids.csv"))
        # take average of steamer and zips talents for our standard talent measure
        steamer_p = pd.read_csv(Path(DATA_PATH, "steamer_pitcher.csv"))
        zips_p = pd.read_csv(Path(DATA_PATH, "zips_pitcher.csv"))
        pitch_proj = pd.merge(
            steamer_p[["Name", "Team", "WAR", "playerid"]],
            zips_p[["WAR", "playerid"]],
            on="playerid",
            suffixes=["_s", "_z"],
            how="outer",
        )
        # drop everyone who isn't on a 40-man roster
        # note: we don't always have projections for the entire 40-man. When
        # that happens we just assume they're replacement level. i.e. WAR = 0
        pitch_proj = pitch_proj[pitch_proj["playerid"].isin(active["pid"])].copy()
        pitch_proj["WAR_P"] = pitch_proj[["WAR_s", "WAR_z"]].mean(axis=1)
        pitch_proj.set_index("playerid", inplace=True)

        steamer_b = pd.read_csv(Path(DATA_PATH, "steamer_batter.csv"))
        zips_b = pd.read_csv(Path(DATA_PATH, "zips_batter.csv"))
        batter_proj = pd.merge(
            steamer_b[["Name", "Team", "WAR", "playerid"]],
            zips_b[["WAR", "playerid"]],
            on="playerid",
            suffixes=["_s", "_z"],
            how="outer",
        )
        batter_proj = batter_proj[batter_proj["playerid"].isin(active["pid"])]
        batter_proj["WAR_B"] = batter_proj[["WAR_s", "WAR_z"]].mean(axis=1)
        batter_proj.set_index("playerid", inplace=True)

        # conduct transactions
        if transactions:
            self._conduct_transactions(pitch_proj, batter_proj, transactions)

        # group all of the WAR projections by team and add them
        pwar_proj = pitch_proj.groupby("Team")["WAR_P"].sum()
        bwar_proj = batter_proj.groupby("Team")["WAR_B"].sum()

        talent = pd.concat([bwar_proj, pwar_proj], axis=1)
        talent["total"] = talent.sum(axis=1)
        talent.reset_index(inplace=True)
        # calculate talent
        if self.talent_measure == "median":
            league_base = np.median(talent["total"])
        elif self.talent_measure == "mean":
            league_base = np.mean(talent["total"])
        talent["talent"] = talent["total"] / league_base - 1

        # merge on division info
        talent["league"] = talent["Team"].map(constants.LEAGUE)
        talent["division"] = talent["Team"].map(constants.DIV)
        return talent

    def _conduct_transactions(self, pitchers, batters, transactions):
        """Update the pitcher and hitter projection DFs with the transactions
        specified

        Parameters
        ----------
        pitchers : pd.DataFrame
            DataFrame with the pitcher projections
        batters : pd.DataFrame
            DataFrame with the batter projections
        transactions : dict
            Dictionary with each transaction. Key: Value pattern is ID: New Team
        """
        assert isinstance(transactions, dict), "Transactions must be dictionary"
        # loop through each transaction and update the player's team
        for _id, team in transactions.items():
            if _id in pitchers.index:
                pitchers.at[_id, "Team"] = team
            elif _id in batters.index:
                batters.at[_id, "Team"] = team
            else:
                msg = f"{_id} is unrecognized player ID"
                raise ValueError(msg)
