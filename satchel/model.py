"""
This moduel contains the heart of satchel. All of the season will be simulated
from this main class
"""
import pandas as pd
import numpy as np
import difflib
import warnings
from . import constants
from .utils import probability_calculations, fetch_fg_projection_data
from .schedules.cache.clear_cache import clear_cache as clear_schedule_cache
from .modelresults import SatchelResults
from .schedules.createschedule import create_schedule, OPENING_DAY, YEAR, FINAL_DAY
from collections import Counter
from pathlib import Path, PosixPath
from typing import Union
from tqdm import tqdm
from datetime import datetime
from pybaseball import standings
from io import StringIO
from typing import Union


CUR_PATH = Path(__file__).resolve().parent
DATA_PATH = Path(CUR_PATH, "data")
SCHEDUEL_PATH = Path(CUR_PATH, "schedules", "schedule2023.csv")
# projections for pitchers and batters
PITCHER_PROJ = Path(DATA_PATH, "pitcherprojections.csv")
BATTER_PROJ = Path(DATA_PATH, "batterprojections.csv")
PROBABILITY_METHOD = "bradley_terry"
ELO_SCALE = 400


class Satchel:
    def __init__(
        self,
        talent_measure: str = "median",
        transactions: dict = None,
        noise: bool = True,
        seed: int = None,
        steamer_p_wt: float = 0.5,
        zips_p_wt: float = 0.5,
        steamer_b_wt: float = 0.5,
        zips_b_wt: float = 0.5,
        schedule: Union[Path, str] = SCHEDUEL_PATH,
        pitcher_proj: Union[PosixPath, str, pd.DataFrame] = "fetch",
        batter_proj: Union[PosixPath, str, pd.DataFrame] = "fetch",
        use_current_results: bool = True,
        war_method: str = "current_pace",
        fg_projections: str = "fangraphsdc",
        cache: bool = True,
    ):
        """
        Main model class

        Parameters
        ----------
        talent_measure: str
            "mean" or "median". Each team's total WAR will be compared to the
            league's `talent_measure` to determine their talent value
        transactions: dict
            Dictionary containing any transactions to include in the simulation.
            The format of the dictionary should be:
            {`player_fangraphs_id`: {"team": `new_team`, "date": `effective_date`}}
        noise: bool
            If true, random noise will be added to each team's talent measure
            during the simulation
        seed : int, float, optional
            seed used for random draws, by default None
        steamer_p_wt: float, optional
            Weight placed on steamer pitcher projections
        zips_p_wt: float, optional
            Weight placed on ZIPs pitcher projections
        steamer_b_wt: float, optional
            Weight placed on steamer batter projections
        zips_b_wt: float, optional
            Weight placed on ZIPs batter projections
        schedule: Path, str, optional
            Path to a CSV with the season schedule
        pitcher_proj: Path, str, optional
            Path to a CSV with pitcher WAR projections suitable for Satchel
        batter_proj: Path, str, optional
            Path to a CSV with batter WAR projections suitable for Satchel
        use_current_results: bool, optional
            If true, Satchel will simulate the season from today's date and add
            those results to each team's current record. This includes using
            both the team's records and the player's stats on the season in the
            talent calculations. If false, Satchel will simulate the full
            season using the provided schedule and pre-season projections
        war_method: str, optional
            Method used for calculating all player's remaining WAR. If
            `only_projections` a player's final WAR will be their WAR to date
            plus their projected WAR multiplied by the fraction of the season
            remaining. If `current_pace`, it will be their current WAR plus
            their projected WAR multiplied by the remaining fraction of the
            season and their relative production rate. The latter is calculated
            by multiplying their projection by the fraction of the season already
            played and dividing their WAR to date by that number
        fg_projections: str, optional
            Which FanGraphs projection to use. Must be in
            `fangraphsdc`, `zips`, `zipsdc`, `steamer`, `atc`, `thebat`, `thebatx`
        cache: bool, optional
            If true, the new scheudle generated will be cached
        """
        if talent_measure.lower() not in ["median", "mean"]:
            raise ValueError("`talent_measure` must be median or mean")
        self.talent_measure = talent_measure
        self.transactions = transactions
        self.current_standings = None

        if not war_method in ["current_pace", "only_projections"]:
            raise ValueError("`war_method must be `current_pace` or `only_projections`")
        self.war_method = war_method

        self.fg_projections = fg_projections

        # if it's before opening day, create the schedule from file. If after,
        # pull the team's current record, then fetch the rest from MLB.com
        # unless the user specifies that they don't want to
        if schedule != SCHEDUEL_PATH and use_current_results:
            warnings.warn(
                (
                    "You have provided a path to a schedule but left"
                    " `use_current_results` = True. As a result, the provided"
                    " schedule will be ignored. To fix this error, set"
                    " `use_current_results`=False"
                )
            )
        today = datetime.today()
        opening_day = datetime.strptime(f"{OPENING_DAY}{YEAR}", "%m%d%Y")
        final_day = datetime.strptime(FINAL_DAY, "%m/%d/%Y")

        if today > opening_day and use_current_results and today < final_day:
            # for running the model after opening day
            # fetch the remaining schedule if it isn't cached
            fmt = "%d%m%Y"
            schedule = Path(
                CUR_PATH, "schedules", "cache", f"schedule{today.strftime(fmt)}.csv"
            )
            if not schedule.exists():
                clear_schedule_cache()  # remove the schedules from previous days
                _return_schedule = False
                if not cache:
                    schedule = None
                    _return_schedule = True
                print("Creating new schedule...")
                sched = create_schedule(
                    year=today.year,
                    start_date=today.strftime("%m%d"),
                    outfile=schedule,
                    _return=_return_schedule,
                )
                if not cache:
                    schedule = StringIO(sched.to_csv(index=False))
            self.current_standings = pd.concat(standings(YEAR))
            self.current_standings["index"] = self.current_standings["Tm"].map(
                constants.NAME_TO_ABBR
            )
            self.current_standings["W"] = self.current_standings["W"].astype(int)
            self.current_standings["L"] = self.current_standings["L"].astype(int)
            self.midseason = True
        else:
            use_current_results = False
            self.midseason = False

        self.schedule = pd.read_csv(schedule, parse_dates=["START DATE"])

        self.teams = constants.DIVS.keys()
        self.random = np.random.default_rng(seed)
        self.noise = noise
        self.seed = seed
        self.steamer_p_wt = steamer_p_wt
        self.zips_p_wt = zips_p_wt
        self.steamer_b_wt = steamer_b_wt
        self.zips_b_wt = zips_b_wt

        # read projection data for calculating talent
        self._set_data(source=pitcher_proj, attr="pitch_proj")
        self._set_data(source=batter_proj, attr="batter_proj")

        self.pitch_proj.rename(columns={"WAR": "WAR_P"}, inplace=True)
        self.pitch_proj.set_index("playerid", inplace=True)

        self.batter_proj.rename(columns={"WAR": "WAR_B"}, inplace=True)
        self.batter_proj.set_index("playerid", inplace=True)

        self.talent, self.st_data = self._calculate_talent(transactions)

    def simulate(
        self,
        n: int = 10000,
        noise: bool = True,
        playoff_func="twelve",
        quiet: bool = False,
        probability_method: str = PROBABILITY_METHOD,
        elo_scale: int = ELO_SCALE,
    ) -> SatchelResults:
        """Run a model simulation n times

        Parameters
        ----------
        n : int, optional
            Number of iterations to run the model for, by default 10000
        noise : bool, optional
            Whether or not to add noise to team talent levels, by default True
        quiet : bool, optional
            If true, suppresses TQDM progress bar when running simulations
        probability_method: str, optional
            Method used to calculate the probability of each team winning the
            game. Accepted options are `bradley_terry` and `elo`.
            Under the Bradley-Terry method, the probability of team A winning is
                P(Team_A) = exp(T_A) / [exp(T_A) + exp(T_B)]
            Under the Elo system, the probability of team A winning is
                P(Team_A) = 1/[1 + 10^((T_A - T_B) / elo_scale)]
            where T_X is the talent level of team X (A or B)
        elo_scalse: int, optional
            The scale parameter used in the Elo probability calculations

        Returns
        -------
        SatchelResults
            Instance of the SatchelResults class.
        """
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
        for i in tqdm(range(n), disable=quiet):
            (
                results,
                playoffs,
                div_winners,
                wc_winners,
                matchups,
                noise,
                full_season,
            ) = self.simseason(
                self.st_data,
                playoff_func=playoff_func,
                current_standings=self.current_standings,
                probability_method=probability_method,
                elo_scale=elo_scale,
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
            self.st_data,
            noise,
            full_seasons,
            self.seed,
        )

    def simseason(
        self,
        data,
        playoff_func,
        current_standings=None,
        probability_method=PROBABILITY_METHOD,
        elo_scale=ELO_SCALE,
    ) -> tuple:
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
        _talent = (
            self.talent[["Team", "final_talent"]].set_index("Team").to_dict("index")
        )
        # add random noise to team talent for the season
        if self.noise:
            _noise = self.random.normal(
                scale=self.talent["base_talent"].std(), size=len(self.teams)
            )
            team_noise = {team: _noise[i] for i, team in enumerate(self.teams)}
            data["a_talent"] += np.array([team_noise[team] for team in data["away"]])
            data["h_talent"] += np.array([team_noise[team] for team in data["home"]])
            for team, value in team_noise.items():
                _talent[team]["final_talent"] += value
        # sim regular season
        home_win_prob = probability_calculations(
            team1_talent=data["h_talent"],
            team2_talent=data["a_talent"],
            probability_method=probability_method,
            elo_scale=elo_scale,
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
        wins = winner.value_counts().reset_index()
        losses = loser.value_counts().reset_index()
        # ensure that teams will always be in the same order
        results = pd.merge(wins, losses, on="index")
        # merge on season-to-date results
        if isinstance(current_standings, pd.DataFrame):
            results = results.merge(current_standings, on="index")
            results["wins"] += results["W"]
            results["losses"] += results["L"]
            results = results.filter(["index", "wins", "losses"], axis="columns")
        results = results.sort_values(
            ["wins", "index"], ascending=False, kind="mergesort"
        )
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
        ) = self.sim_playoff(results, _talent, playoff_func=playoff_func)
        # column for season result. This is the best they do in the season
        results["season_result"] = np.where(
            results["Team"].isin(div_winners["Team"]),
            "Division Champ",
            np.where(
                results["Team"].isin(wc_winners["Team"]), "Wild Card", "Missed Playoffs"
            ),
        )
        results["season_result"] = np.where(
            results["Team"].isin(cs_winners), "Win League", results["season_result"]
        )
        results["season_result"] = np.where(
            results["Team"] == final_res["ws"],
            "Win World Series",
            results["season_result"],
        )
        # flags for individual season results. Need this because season_result
        # won't show if they won the wild card, division, etc. if they
        # reach a higher achievement
        results["wild_card"] = np.where(results["Team"].isin(wc_winners["Team"]), 1, 0)
        results["won_division"] = np.where(
            results["Team"].isin(div_winners["Team"]), 1, 0
        )
        results["won_league"] = np.where(results["Team"].isin(cs_winners), 1, 0)
        results["won_ws"] = np.where(results["Team"] == final_res["ws"], 1, 0)
        return results, final_res, div_winners, wc_winners, matchups, team_noise, data

    def sim_playoff(
        self,
        results,
        talent,
        n_divwinners=1,
        n_wildcard=3,
        playoff_func="twelve",
        probability_method=PROBABILITY_METHOD,
        elo_scale=ELO_SCALE,
    ):
        """Run the playoff simulation.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with the results of the regular season
        talent : pd.DataFrame
            DataFrame with the team talent for the season
        n_divwinners : int, optional
            number of division winners, by default 1
        n_wildcard : int, optional
            Number of wild card winners, by default 2
        playoff_func: str, optional
            String to indicate which play off function is used. Right now must
            be either 'twelve' for a 12 team playoff, or 'ten' for 10 team.

        Returns
        -------
        tuple
            Tuple with league results, leage champions, division and wild card
            winners, and all of the post season matchups
        """

        div_winners = (
            results.groupby(["league", "division"])
            .apply(pd.DataFrame.nlargest, n=n_divwinners, columns="wins")
            .sort_values("wins", ascending=False)
        )
        # take out division winners and pull the top remaining teams 4 wild card
        wc = results[~results["Team"].isin(div_winners["Team"])]
        wc_winners = (
            wc.groupby("league")
            .apply(pd.DataFrame.nlargest, n=n_wildcard, columns="wins")
            .sort_values("wins", ascending=False)
        )
        # determine which playoff format will be used
        _playoff_func = self._twelve_team_playoff
        if playoff_func == "ten":
            _playoff_func = self._ten_team_playoff
        # simulate all the rounds
        nlres, matchups = _playoff_func(
            wc_winners.loc["NL"], div_winners.loc["NL"], talent, "NL", {}
        )
        alres, matchups = _playoff_func(
            wc_winners.loc["AL"], div_winners.loc["AL"], talent, "AL", matchups
        )
        # world series winner
        matchups["World Series"] = "-".join(sorted([nlres["cs"], alres["cs"]]))
        champ = self._sim_round(
            [nlres["cs"], alres["cs"]],
            talent,
            7,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups["WS Winner"] = champ
        return (
            {"nl": nlres, "al": alres, "ws": champ},
            [alres["cs"], nlres["cs"]],
            div_winners,
            wc_winners,
            matchups,
        )

    def matchup(
        self,
        team1: str,
        team2: str,
        probability_method: str = PROBABILITY_METHOD,
        elo_scale: int = ELO_SCALE,
    ) -> float:
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
        team1_talent = self.talent["base_talent"][self.talent["Team"] == team1].values
        team2_talent = self.talent["base_talent"][self.talent["Team"] == team2].values
        # team1_prob = np.exp(team1_talent) / (
        #     np.exp(team1_talent) + np.exp(team2_talent)
        # )
        team1_prob = probability_calculations(
            team1_talent=team1_talent,
            team2_talent=team2_talent,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )

        return team1_prob, 1 - team1_prob

    ####### Private methods #######

    def _set_data(self, source, attr):
        """
        Private method for reading and accounting for projection data

        Parameters
        ----------
        source : str
            Either a string indicating the projection data should be fetched,
            a string or path leading to a CSV file that can be read, or a DataFrame
            with the data already in it
        attr : str
            Which projections attribute to set. Either `pitch_proj` or `batter_proj`
        """
        if source == "fetch":
            setattr(
                self,
                attr,
                fetch_fg_projection_data(
                    attr[:3], self.fg_projections, datetime.today()
                ),
            )
        elif isinstance(source, pd.DataFrame):
            setattr(self, attr, source)
        elif isinstance(source, str) or isinstance(source, PosixPath):
            setattr(self, attr, pd.read_csv(source))
        else:
            raise ValueError("Projections must be from a string, path, or dataframe.")

    def _calculate_talent(self, transactions=None, pitcher_wt=1, batter_wt=1):
        """Private method used to calculate each team's talent level by taking
        the depth chart projections from FanGraphs

        Parameters
        ----------
        transactions : dict, optional
            Dictionary containing transaction information.

        Returns
        -------
        pd.DataFrame
            DataFrame containing talent levels for each team.
        """
        # group all of the WAR projections by team and add them
        pwar_proj = self.pitch_proj.groupby("Team")["WAR_P"].sum()
        bwar_proj = self.batter_proj.groupby("Team")["WAR_B"].sum()

        talent = pd.concat([bwar_proj, pwar_proj], axis=1)
        # allow users to place more weight on pitchers or hitters, equally
        # weighted by default
        talent["total"] = talent["WAR_P"] * pitcher_wt + talent["WAR_B"] * batter_wt
        talent.reset_index(inplace=True)
        # calculate baseline talent
        if self.talent_measure == "median":
            league_base = np.median(talent["total"])
        elif self.talent_measure == "mean":
            league_base = np.mean(talent["total"])
        # baseline talent before any transactions
        talent["base_talent"] = talent["total"] / league_base - 1

        # merge on division info
        talent["league"] = talent["Team"].map(constants.LEAGUE)
        talent["division"] = talent["Team"].map(constants.DIV)

        # merge together schedule and talent data
        st_data = pd.merge(
            self.schedule,
            talent[["Team", "total"]],
            left_on="away",
            right_on="Team",
        )
        st_data = pd.merge(
            st_data, talent[["Team", "total"]], left_on="home", right_on="Team"
        )
        # clean up data after merge
        st_data.drop(["Team_x", "Team_y"], axis=1, inplace=True)
        st_data.rename(
            columns={"total_x": "away_total", "total_y": "home_total"}, inplace=True
        )

        # conduct transactions
        if transactions:
            talent = self._conduct_transactions(
                self.pitch_proj, self.batter_proj, transactions, st_data, talent
            )
            talent["final_talent"] = talent["final_total"] / league_base - 1
        else:
            talent["final_talent"] = talent["base_talent"]

        # calculate talent levels used for simulations
        st_data["away_talent"] = st_data["away_total"] / league_base - 1
        st_data["home_talent"] = st_data["home_total"] / league_base - 1

        return talent, st_data

    def _conduct_transactions(self, pitchers, batters, transactions, st_data, talent):
        """Update the pitcher and hitter projection DFs with the transactions
        specified. Format should be:
        {player_id: {'date': 'YYYY-MM-DD', 'team': new_team}}

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
        last_date = datetime.strptime("1900-01-01", "%Y-%m-%d")
        for _id, info in transactions.items():
            # verify the team exists
            team = info["team"].upper()
            # check that it's a valid team name, or no team at all
            if team not in self.teams and team != "":
                close = difflib.get_close_matches(team, self.teams)
                msg = f"{team} is not a valid team. Close matches are: {close}"
                raise ValueError(msg)
            # find the old team and WAR of player being moved
            if _id in pitchers.index:
                old_team = pitchers.at[_id, "Team"]
                war = pitchers.at[_id, "WAR_P"]
            elif _id in batters.index:
                old_team = batters.at[_id, "Team"]
                war = batters.at[_id, "WAR_B"]
            else:
                msg = f"{_id} is an unrecognized player ID"
                raise ValueError(msg)
            # update the talent for each each team involved
            st_data["away_total"] = np.where(
                (st_data["START DATE"] >= info["date"]) & (st_data["away"] == team),
                st_data["away_total"] + war,
                st_data["away_total"],
            )
            st_data["home_total"] = np.where(
                (st_data["START DATE"] >= info["date"]) & (st_data["home"] == team),
                st_data["home_total"] + war,
                st_data["home_total"],
            )
            # remove the player's WAR from their old team
            st_data["away_total"] = np.where(
                (st_data["START DATE"] >= info["date"]) & (st_data["away"] == old_team),
                st_data["away_total"] - war,
                st_data["away_total"],
            )
            st_data["home_total"] = np.where(
                (st_data["START DATE"] >= info["date"]) & (st_data["home"] == old_team),
                st_data["home_total"] - war,
                st_data["home_total"],
            )
            # update date of last transaction
            if datetime.strptime(info["date"], "%Y-%m-%d") > last_date:
                last_date = datetime.strptime(info["date"], "%Y-%m-%d")

        # find the final talent for each team after all the transactions. This
        # will be used for the playoff simulations
        finaldf = (
            st_data[st_data["START DATE"] >= last_date]
            .groupby("away")["away_total"]
            .mean()
            .reset_index()
            .rename({"away": "Team", "away_total": "final_total"}, axis=1)
        )
        talent = talent.merge(finaldf, on="Team")

        return talent

    def _sim_round(self, teams, talent, n_games, probability_method, elo_scale):
        """
        Simulate a playoff round with n_games
        """
        team1 = 0
        team2 = 0
        team1_win_prob = probability_calculations(
            team1_talent=talent[teams[0]]["final_talent"],
            team2_talent=talent[teams[1]]["final_talent"],
            probability_method=probability_method,
            elo_scale=elo_scale,
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

    # playoff round functions
    def _ten_team_playoff(
        self,
        wc_winners,
        div_winners,
        talent,
        league,
        matchups,
        probability_method=PROBABILITY_METHOD,
        elo_scale=ELO_SCALE,
    ):
        """
        Simulate a ten team post season
        """
        # sort and join the teams so that all match ups count the same
        wc = "-".join(sorted([wc_winners["Team"].iloc[0], wc_winners["Team"].iloc[1]]))
        matchups[f"{league} Wild Card"] = wc
        wc_winner = self._sim_round(
            [wc_winners["Team"].iloc[0], wc_winners["Team"].iloc[1]],
            talent,
            1,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league} WC Champ"] = wc_winner
        ds1 = "-".join(sorted([wc_winner, div_winners["Team"].iloc[0]]))
        matchups[f"{league}DS 1"] = ds1
        div_rd1 = self._sim_round(
            [wc_winner, div_winners["Team"].iloc[0]],
            talent,
            5,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league}DS 1 Champ"] = div_rd1
        ds2 = "-".join(
            sorted([div_winners["Team"].iloc[1], div_winners["Team"].iloc[2]])
        )
        matchups[f"{league}DS 2"] = ds2
        div_rd2 = self._sim_round(
            [div_winners["Team"].iloc[1], div_winners["Team"].iloc[2]],
            talent,
            5,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league}DS 2 Champ"] = div_rd2
        matchups[f"{league}CS"] = "-".join(sorted([div_rd1, div_rd2]))
        cs = self._sim_round(
            [div_rd1, div_rd2],
            talent,
            7,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league} Champ"] = cs
        return (
            {"wc": wc_winner, "div_rd1": div_rd1, "div_rd2": div_rd2, "cs": cs},
            matchups,
        )

    def _twelve_team_playoff(
        self,
        wc_winners,
        div_winners,
        talent,
        league,
        matchups,
        probability_method=PROBABILITY_METHOD,
        elo_scale=ELO_SCALE,
    ):
        """
        12 team playoff with the following bracket in each league:
        Round 1:
            - Top two division winners have a bye
            - Third division winner plays the lowest ranked wild card winner (1)
            - Top two wild card winners play each other (2)
        Round 2:
            - Top seed plays the winner of match up (1)
            - Second seed plays the winner of match up (2)
        Round 3:
            - The winners of the above round play each other
        """
        # Round 1
        # top two wild card seeds
        wc1 = "-".join(sorted([wc_winners["Team"].iloc[0], wc_winners["Team"].iloc[1]]))
        matchups[f"{league} Wild Card 1"] = wc1
        wc1_winner = self._sim_round(
            [wc_winners["Team"].iloc[0], wc_winners["Team"].iloc[1]],
            talent,
            3,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league} WC 1 Champ"] = wc1_winner
        # bottom wild card seed and bottom division winner
        wc2 = "-".join(
            sorted([wc_winners["Team"].iloc[2], div_winners["Team"].iloc[2]])
        )
        matchups[f"{league} Wild Card 2"] = wc2
        wc2_winner = self._sim_round(
            [wc_winners["Team"].iloc[2], div_winners["Team"].iloc[2]],
            talent,
            3,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"WC 2 Champ"] = wc2_winner
        # Round 2
        # top seeded division winner vs. WC 1 winner
        div1 = "-".join(sorted([wc1_winner, div_winners["Team"].iloc[0]]))
        matchups[f"{league}DS 1"] = div1
        div1_winner = self._sim_round(
            [wc1_winner, div_winners["Team"].iloc[0]],
            talent,
            7,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league}DS 1 Champ"] = div1_winner
        # second seed division winner vs. WC 2 winner
        div2 = "-".join(sorted([wc2_winner, div_winners["Team"].iloc[1]]))
        matchups[f"{league}DS 2"] = div2
        div2_winner = self._sim_round(
            [wc2_winner, div_winners["Team"].iloc[1]],
            talent,
            7,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league}DS 2 Champ"] = div2_winner
        # Round 3
        # league championship
        matchups[f"{league}CS"] = "-".join(sorted([div1_winner, div2_winner]))
        cs = self._sim_round(
            [div1_winner, div2_winner],
            talent,
            7,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league} Champ"] = cs

        return (
            {
                "wc1": wc1_winner,
                "wc2": wc2_winner,
                "div_rd1": div1_winner,
                "div_rd2": div2_winner,
                "cs": cs,
            },
            matchups,
        )

    # def _prep_talent_data(self, batter_stats, pitcher_stats, war_method):
    #     def process(data, stats, remaining, method):
    #         _data = pd.merge(
    #             data, stats, on="playerid", suffixes=["_proj", "_cur"], how="outer"
    #         )
    #         # some players may be in the projections, but not the current data
    #         # or vice versa. For these cases, replace missing values with the
    #         # value we do have
    #         war_proj = _data["WAR_proj"]
    #         war_cur = _data["WAR_cur"]
    #         _data["WAR_cur"].fillna(war_proj, inplace=True)
    #         _data["WAR_proj"].fillna(war_cur, inplace=True)
    #         del war_proj, war_cur
    #         if method == "only_projections":
    #             _data["WAR"] = (_data["WAR_proj"] * (remaining)) + _data["WAR_cur"]
    #         elif method == "current_pace":
    #             # at this point in the season, they should have some fraction of
    #             # their projected war.
    #             hypothetical = _data["WAR_proj"] * (1 - remaining)
    #             # the rate at which they're under/over performing
    #             production_rate = (_data["WAR_cur"] / hypothetical).fillna(1)
    #             # final WAR is their current WAR plus the fraction remaining of
    #             # their projected WAR if they keep up with their current pace
    #             _data["WAR"] = _data["WAR_cur"] + (
    #                 _data["WAR_proj"] * remaining * production_rate
    #             )
    #         return _data

    #     # calculate how much of the season has been played. This fraction will
    #     # be used to weight the projections and current results when getting
    #     # final WAR used in the talent calculations
    #     games_played = (
    #         self.current_standings["W"] + self.current_standings["L"]
    #     ).mean()
    #     remaining = (162 - games_played) / 162

    #     _batter = process(self.batter_proj, batter_stats, remaining, war_method)
    #     _pitcher = process(self.pitch_proj, pitcher_stats, remaining, war_method)
    #     self.batter_proj = _batter.drop(columns=["WAR_proj", "WAR_cur"])
    #     self.pitch_proj = _pitcher.drop(columns=["WAR_proj", "WAR_cur"])
