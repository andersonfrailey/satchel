"""
This moduel contains the heart of satchel. All of the season will be simulated
from this main class
"""

import difflib
import warnings
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import constants
from .modelresults import SatchelResults
from .schedules.cache.clear_cache import clear_cache as clear_schedule_cache

# TODO: save all these dates by year so they may be called based on simulation
from .schedules.createschedule import (
    ALL_STAR_BREAK,
    FINAL_DAY,
    OPENING_DAY,
    YEAR,
    create_schedule,
)
from .standings import standings
from .utils import fetch_fg_projection_data, probability_calculations

CUR_PATH = Path(__file__).resolve().parent
DATA_PATH = Path(CUR_PATH, "data")
SCHEDUEL_PATH = Path(CUR_PATH, "schedules")
# projections for pitchers and batters
PITCHER_PROJ = Path(DATA_PATH, "pitcherprojections.csv")
BATTER_PROJ = Path(DATA_PATH, "batterprojections.csv")
PROBABILITY_METHOD = "bradley_terry"
ELO_SCALE = 400


class Satchel:
    pitch_proj: pd.DataFrame
    batter_proj: pd.DataFrame

    def __init__(
        self,
        talent_measure: str = "median",
        transactions: dict | None = None,
        noise: bool = True,
        seed: int | None = None,
        steamer_p_wt: float = 0.5,
        zips_p_wt: float = 0.5,
        steamer_b_wt: float = 0.5,
        zips_b_wt: float = 0.5,
        schedule: Path | str | StringIO | None = SCHEDUEL_PATH,
        pitcher_proj: PosixPath | str | pd.DataFrame = "fetch",
        batter_proj: PosixPath | str | pd.DataFrame = "fetch",
        use_current_results: bool = True,
        war_method: str = "current_pace",
        fg_projections: str = "fangraphsdc",
        year: int = YEAR,
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

        if war_method not in ["current_pace", "only_projections"]:
            raise ValueError("`war_method must be `current_pace` or `only_projections`")
        self.war_method = war_method

        self.fg_projections = fg_projections
        self.all_star_break = ALL_STAR_BREAK

        # if it's before opening day, create the schedule from file. If after,
        # pull the team's current record, then fetch the rest from MLB.com
        # unless the user specifies that they don't want to
        if not isinstance(schedule, pd.DataFrame):
            if schedule != SCHEDUEL_PATH and use_current_results:
                warnings.warn(
                    (
                        "You have provided a path to a schedule but left"
                        " `use_current_results` = True. As a result, the provided"
                        " schedule will be ignored. To fix this warning, set"
                        " `use_current_results`=False"
                    )
                )
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        opening_day = datetime.strptime(f"{OPENING_DAY}{YEAR}", "%m%d%Y")
        final_day = datetime.strptime(f"{FINAL_DAY}{YEAR}", "%m%d%Y")

        if today >= opening_day and use_current_results and today <= final_day:
            # for running the model after opening day
            # fetch the remaining schedule if it isn't cached
            fmt = "%d%m%Y"
            schedule = Path(
                CUR_PATH, "schedules", "cache", f"schedule{today.strftime(fmt)}.csv"
            )
            if not schedule.exists():
                clear_schedule_cache()  # remove the schedules from previous days
                if not cache:
                    schedule = None
                print("Creating new schedule...")
                sched = create_schedule(
                    year=today.year,
                    start_date=today.strftime("%m%d"),
                    outfile=schedule,
                )
                if not cache:
                    schedule = StringIO(sched.to_csv(index=False))
            self.current_standings = standings(YEAR)
            self.current_standings["index"] = self.current_standings["Tm"].map(
                constants.NAME_TO_ABBR
            )
            self.current_standings["W"] = self.current_standings["W"].astype(int)
            self.current_standings["L"] = self.current_standings["L"].astype(int)
            self.midseason = True
        else:
            self.midseason = False
            schedule = Path(SCHEDUEL_PATH, f"schedule{year}.csv")

        self.schedule = pd.read_csv(schedule, parse_dates=["START DATE"])  # type:ignore
        # name change for Oakland
        self.schedule["home"] = np.where(
            self.schedule["home"] == "OAK", "ATH", self.schedule["home"]
        )
        self.schedule["away"] = np.where(
            self.schedule["away"] == "OAK", "ATH", self.schedule["away"]
        )

        self.teams = constants.DIVS.keys()
        self.random = np.random.default_rng(seed)
        self.noise = noise
        self.seed = seed
        self.steamer_p_wt = steamer_p_wt
        self.zips_p_wt = zips_p_wt
        self.steamer_b_wt = steamer_b_wt
        self.zips_b_wt = zips_b_wt

        # read projection data for calculating talent
        self._set_data(source=pitcher_proj, attr="pitch_proj", stats="pit")
        self._set_data(source=batter_proj, attr="batter_proj", stats="bat")

        self.pitch_proj.rename(columns={"WAR": "WAR_P"}, inplace=True)
        self.pitch_proj.set_index("playerid", inplace=True)

        self.batter_proj.rename(columns={"WAR": "WAR_B"}, inplace=True)
        self.batter_proj.set_index("playerid", inplace=True)

        self.talent, self.st_data = self._calculate_talent(transactions)
        self.st_data.sort_values("START DATE", inplace=True, ignore_index=True)

        # track number of ties by type
        self.two_way_ties = 0
        self.three_way_ties = 0
        self.four_way_ties = 0

    def simulate(
        self,
        n: int = 20000,
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
        all_wins_to_date = []
        for i in tqdm(range(n), disable=quiet):
            (
                results,
                playoffs,
                div_winners,
                wc_winners,
                matchups,
                noise,
                full_season,
                wins_to_date,
            ) = self.simseason(
                self.st_data.copy(),
                playoff_func=playoff_func,
                current_standings=self.current_standings,
                probability_method=probability_method,
                elo_scale=elo_scale,
            )
            ws_counter.update([playoffs["ws"]])
            div_counter.update(div_winners)
            league_counter.update([playoffs["nl"]["cs"]])
            league_counter.update([playoffs["al"]["cs"]])
            wc_counter.update(wc_winners)
            playoff_counter.update(wc_winners)
            playoff_counter.update(div_winners)
            results["sim"] = i
            all_results.append(results)
            all_matchups.append(matchups)
            all_noise.append(noise)
            full_seasons.append(full_season)
            all_wins_to_date.append(wins_to_date)

        return SatchelResults(
            ws_counter=ws_counter,
            league_counter=league_counter,
            div_counter=div_counter,
            wc_counter=wc_counter,
            playoff_counter=playoff_counter,
            results_df=pd.concat(all_results),
            playoff_matchups=pd.DataFrame(all_matchups),
            base_talent=self.talent,
            n=n,
            trades=self.transactions,
            schedule=self.schedule,
            merged_schedule=self.st_data,
            noise=all_noise,
            full_seasons=full_seasons,
            seed=self.seed,
            fg_projections=self.fg_projections,
            two_way_ties=self.two_way_ties,
            three_way_ties=self.three_way_ties,
            four_way_ties=self.four_way_ties,
            date=datetime.strftime(datetime.today(), "%m-%d-%Y"),
            season_wins_to_date=all_wins_to_date,
        )

    def simseason(
        self,
        data,
        playoff_func,
        current_standings=None,
        probability_method=PROBABILITY_METHOD,
        elo_scale=ELO_SCALE,
    ) -> tuple[
        pd.DataFrame, dict, list[str], list[str], dict, dict, pd.DataFrame, pd.DataFrame
    ]:
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
        team_noise = {team: 0 for team in self.teams}
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
        data["home_win_prob"] = home_win_prob
        probs = self.random.random(len(data))
        data["probability_draw"] = probs
        winner = pd.Series(
            np.where(home_win_prob >= probs, data["home"], data["away"]), name="wins"
        )
        loser = pd.Series(
            np.where(home_win_prob >= probs, data["away"], data["home"]), name="losses"
        )
        data["winner"] = winner
        data["loser"] = loser
        # get cumulative wins to date for each team
        data["one"] = 1
        data["wins_to_date"] = data.groupby("winner")["one"].cumsum()
        data.drop(columns=["one"], inplace=True)
        wins = winner.value_counts().reset_index()
        wins.rename(columns={"wins": "index", "count": "wins"}, inplace=True)
        losses = loser.value_counts().reset_index()
        losses.rename(columns={"losses": "index", "count": "losses"}, inplace=True)
        # find wins to date for each team
        home = data[["START DATE", "home"]].copy().rename(columns={"home": "team"})
        away = data[["START DATE", "away"]].copy().rename(columns={"away": "team"})
        wins_to_date = pd.concat([home, away])
        wins_to_date = wins_to_date.merge(
            data[["START DATE", "winner", "wins_to_date"]],
            left_on=["START DATE", "team"],
            right_on=["START DATE", "winner"],
            how="left",
        )
        wins_to_date.sort_values("START DATE", inplace=True, ignore_index=True)
        wins_to_date["wins_to_date"] = (
            wins_to_date.groupby("team")["wins_to_date"].ffill().fillna(0)
        )
        wins_to_date.drop(columns="winner", inplace=True)
        # get game number
        wins_to_date["one"] = 1
        wins_to_date["game"] = wins_to_date.groupby("team")["one"].cumsum()
        wins_to_date.drop(columns=["one"], inplace=True)
        # count up head-to-head losses. resulting dict has key: value pair: (winner, loser): h2h wins
        h2h = data.groupby(["winner", "loser"]).size().to_dict()
        # outer merge because during simulations late in the season not all teams
        # will appear in both wins and losses if using the current standings.
        # Some will win or lose all of their games, leaving them out of the
        # other DataFrame
        results = pd.merge(wins, losses, on="index", how="outer").fillna(0)
        # merge on season-to-date results
        if isinstance(current_standings, pd.DataFrame):
            results = results.merge(current_standings, on="index", how="outer")
            # near end of season, not everyone will have simulated wins.
            # fill that in with zero
            results.fillna(0, inplace=True)
            results["wins"] += results["W"]
            results["losses"] += results["L"]
            results = results.filter(["index", "wins", "losses"], axis="columns")
        assert results.shape[0] == len(self.teams)
        # use merge sort to ensure that teams will always be in the same order
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
        ) = self.sim_playoff(results, h2h, data, _talent, playoff_func=playoff_func)
        # column for season result. This is the best they do in the season
        results["season_result"] = np.where(
            results["Team"].isin(div_winners),
            "Division Champ",
            np.where(results["Team"].isin(wc_winners), "Wild Card", "Missed Playoffs"),
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
        results["wild_card"] = np.where(results["Team"].isin(wc_winners), 1, 0)
        results["won_division"] = np.where(results["Team"].isin(div_winners), 1, 0)
        results["won_league"] = np.where(results["Team"].isin(cs_winners), 1, 0)
        results["won_ws"] = np.where(results["Team"] == final_res["ws"], 1, 0)
        return (
            results,
            final_res,
            div_winners,
            wc_winners,
            matchups,
            team_noise,
            data,
            wins_to_date,
        )

    def sim_playoff(
        self,
        results: pd.DataFrame,
        h2h: dict[tuple[str, str], int],
        data: pd.DataFrame,
        talent: dict,
        n_wildcard: int = 3,
        playoff_func: str = "twelve",
        probability_method: str = PROBABILITY_METHOD,
        elo_scale: int = ELO_SCALE,
    ):
        """Run the playoff simulation.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with the results of the regular season
        h2h : dict
            Head-to-head record dict with (winner, loser): wins format
        data : pd.DataFrame
            DataFrame with season game results for tiebreaker calculations
        talent : dict
            DataFrame with the team talent for the season
        n_wildcard : int, optional
            Number of wild card winners, by default 3
        playoff_func: str, optional
            String to indicate which play off function is used. Right now must
            be either 'twelve' for a 12 team playoff, or 'ten' for 10 team.

        Returns
        -------
        tuple
            Tuple with league results, leage champions, division and wild card
            winners, and all of the post season matchups
        """
        # Select playoff teams for each league using tiebreaker rules
        al_div_winners, al_wc_winners = self._select_playoff_teams(
            results, h2h, data, "AL", n_wild_cards=n_wildcard
        )
        nl_div_winners, nl_wc_winners = self._select_playoff_teams(
            results, h2h, data, "NL", n_wild_cards=n_wildcard
        )

        # Combine into lists for return value compatibility
        div_winners = al_div_winners + nl_div_winners
        wc_winners = al_wc_winners + nl_wc_winners

        # determine which playoff format will be used
        _playoff_func = self._twelve_team_playoff
        if playoff_func == "ten":
            _playoff_func = self._ten_team_playoff

        # simulate all the rounds (now passing seeded lists instead of DataFrames)
        nlres, matchups = _playoff_func(nl_wc_winners, nl_div_winners, talent, "NL", {})
        alres, matchups = _playoff_func(
            al_wc_winners, al_div_winners, talent, "AL", matchups
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
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Calculate the probability of two teams winning when they play each other

        Parameters
        ----------
        team1 : str
            First team in the matchup
        team2 : str
            Second team in the matchup

        Returns
        -------
        tuple[float, float]
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
        team1_prob = probability_calculations(
            team1_talent=team1_talent,
            team2_talent=team2_talent,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )

        return team1_prob, 1 - team1_prob

    ####### Private methods #######

    def _set_data(self, source, attr, stats):
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
        if isinstance(source, pd.DataFrame):
            data = source
        elif source == "fetch":
            data = fetch_fg_projection_data(
                stats=stats,
                fg_projection=self.fg_projections,
                date=datetime.today(),
            )
        elif isinstance(source, str) or isinstance(source, PosixPath):
            data = pd.read_csv(source)
        else:
            raise ValueError("Projections must be from a string, path, or dataframe.")

        # account for Oakland move
        data["Team"] = np.where(data["Team"] == "OAK", "ATH", data["Team"])
        setattr(self, attr, data)

    def _calculate_talent(
        self, transactions=None, pitcher_wt=1, batter_wt=1
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        else:
            raise ValueError("leage_base must be `mean` or `median`")
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
        wc_winners: list[str],
        div_winners: list[str],
        talent,
        league,
        matchups,
        probability_method=PROBABILITY_METHOD,
        elo_scale=ELO_SCALE,
    ):
        """
        Simulate a ten team post season.

        Parameters
        ----------
        wc_winners : list[str]
            List of wild card teams, seeded (1st seed first)
        div_winners : list[str]
            List of division winners, seeded (1st seed first)
        """
        # sort and join the teams so that all match ups count the same
        wc = "-".join(sorted([wc_winners[0], wc_winners[1]]))
        matchups[f"{league} Wild Card"] = wc
        wc_winner = self._sim_round(
            [wc_winners[0], wc_winners[1]],
            talent,
            1,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league} WC Champ"] = wc_winner
        ds1 = "-".join(sorted([wc_winner, div_winners[0]]))
        matchups[f"{league}DS 1"] = ds1
        div_rd1 = self._sim_round(
            [wc_winner, div_winners[0]],
            talent,
            5,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league}DS 1 Champ"] = div_rd1
        ds2 = "-".join(sorted([div_winners[1], div_winners[2]]))
        matchups[f"{league}DS 2"] = ds2
        div_rd2 = self._sim_round(
            [div_winners[1], div_winners[2]],
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
        wc_winners: list[str],
        div_winners: list[str],
        talent,
        league: str,
        matchups: dict,
        probability_method: str = PROBABILITY_METHOD,
        elo_scale: int = ELO_SCALE,
    ) -> tuple[dict[str, str], dict[str, str]]:
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

        Parameters
        ----------
        wc_winners : list[str]
            List of wild card teams, seeded (1st seed first)
        div_winners : list[str]
            List of division winners, seeded (1st seed first)
        """
        # Wild Card Round
        # Bottom WC team vs bottom division winner (6 and 4 seed)
        wc1 = "-".join(sorted([wc_winners[2], div_winners[2]]))
        matchups[f"{league} Wild Card 1"] = wc1
        wc1_winner = self._sim_round(
            [wc_winners[2], div_winners[2]],
            talent,
            3,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league} WC 1 Champ"] = wc1_winner
        # top two WC teams (5 and 4 seed)
        wc2 = "-".join(sorted([wc_winners[0], wc_winners[1]]))
        matchups[f"{league} Wild Card 2"] = wc2
        wc2_winner = self._sim_round(
            [wc_winners[0], wc_winners[1]],
            talent,
            3,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league }WC 2 Champ"] = wc2_winner
        # Division Series Round
        # Second seeded division winner vs. WC 1 winner
        div1 = "-".join(sorted([wc1_winner, div_winners[1]]))
        matchups[f"{league}DS 1"] = div1
        div1_winner = self._sim_round(
            [wc1_winner, div_winners[1]],
            talent,
            7,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league}DS 1 Champ"] = div1_winner
        # Top seed division winner vs. WC 2 winner
        div2 = "-".join(sorted([wc2_winner, div_winners[0]]))
        matchups[f"{league}DS 2"] = div2
        div2_winner = self._sim_round(
            [wc2_winner, div_winners[0]],
            talent,
            7,
            probability_method=probability_method,
            elo_scale=elo_scale,
        )
        matchups[f"{league}DS 2 Champ"] = div2_winner
        # Championship Series
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

    def _get_intradiv_record(self, team: str, data: pd.DataFrame) -> dict:
        """
        Compute intradivision record for a single team.

        Parameters
        ----------
        team : str
            Team abbreviation
        data : pd.DataFrame
            DataFrame with season game results (must have 'winner' and 'loser' columns)

        Returns
        -------
        dict
            Dictionary with 'wins', 'losses', and 'pct' keys
        """
        team_div = constants.DIVS[team]
        div_teams = [
            t for t, d in constants.DIVS.items() if d == team_div and t != team
        ]

        # Games where team played against division opponents
        team_in_game = (data["winner"] == team) | (data["loser"] == team)
        opponent_in_div = data["winner"].isin(div_teams) | data["loser"].isin(div_teams)
        div_games = data[team_in_game & opponent_in_div]

        wins = (div_games["winner"] == team).sum()
        losses = (div_games["loser"] == team).sum()
        pct = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        return {"wins": wins, "losses": losses, "pct": pct}

    def _get_intraleague_record(self, team: str, data: pd.DataFrame) -> dict:
        """
        Compute intraleague record for a single team.

        Parameters
        ----------
        team : str
            Team abbreviation
        data : pd.DataFrame
            DataFrame with season game results (must have 'winner' and 'loser' columns)

        Returns
        -------
        dict
            Dictionary with 'wins', 'losses', and 'pct' keys
        """
        team_league = constants.LEAGUE[team]
        league_teams = [
            t for t, lg in constants.LEAGUE.items() if lg == team_league and t != team
        ]

        # Games where team played against league opponents
        team_in_game = (data["winner"] == team) | (data["loser"] == team)
        opponent_in_league = data["winner"].isin(league_teams) | data["loser"].isin(
            league_teams
        )
        league_games = data[team_in_game & opponent_in_league]

        wins = (league_games["winner"] == team).sum()
        losses = (league_games["loser"] == team).sum()
        pct = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        return {"wins": wins, "losses": losses, "pct": pct}

    def _get_last_half_intraleague_record(self, team: str, data: pd.DataFrame) -> dict:
        """
        Compute last half (post All-Star break) intraleague record for a single team.

        Parameters
        ----------
        team : str
            Team abbreviation
        data : pd.DataFrame
            DataFrame with season game results (must have 'winner', 'loser', and 'START DATE' columns)

        Returns
        -------
        dict
            Dictionary with 'wins', 'losses', and 'pct' keys
        """
        team_league = constants.LEAGUE[team]
        league_teams = [
            t for t, lg in constants.LEAGUE.items() if lg == team_league and t != team
        ]

        # Filter to games after All-Star break
        second_half = data[data["START DATE"] >= self.all_star_break]

        # Games where team played against league opponents in second half
        team_in_game = (second_half["winner"] == team) | (second_half["loser"] == team)
        opponent_in_league = second_half["winner"].isin(league_teams) | second_half[
            "loser"
        ].isin(league_teams)
        league_games = second_half[team_in_game & opponent_in_league]

        wins = (league_games["winner"] == team).sum()
        losses = (league_games["loser"] == team).sum()
        pct = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        return {"wins": wins, "losses": losses, "pct": pct}

    def _get_intraleague_games_chronological(
        self, team: str, data: pd.DataFrame, exclude_teams: list[str] | None = None
    ) -> list[tuple]:
        """
        Get chronologically ordered list of intraleague games for a team.
        Used for the 'plus one' tiebreaker.

        Parameters
        ----------
        team : str
            Team abbreviation
        data : pd.DataFrame
            DataFrame with season game results
        exclude_teams : list[str], optional
            Teams to exclude from the list (games between tied clubs are skipped)

        Returns
        -------
        list[tuple]
            List of (date, opponent, won: bool) tuples, ordered chronologically
        """
        if exclude_teams is None:
            exclude_teams = []

        team_league = constants.LEAGUE[team]
        league_teams = [
            t
            for t, lg in constants.LEAGUE.items()
            if lg == team_league and t != team and t not in exclude_teams
        ]

        # Games where team played against league opponents (excluding tied teams)
        team_in_game = (data["winner"] == team) | (data["loser"] == team)
        opponent_in_league = data["winner"].isin(league_teams) | data["loser"].isin(
            league_teams
        )
        league_games = data[team_in_game & opponent_in_league].copy()

        # Sort by date
        league_games = league_games.sort_values("START DATE")

        # Build list of (date, opponent, won)
        games = []
        for _, row in league_games.iterrows():
            if row["winner"] == team:
                opponent = row["loser"]
                won = True
            else:
                opponent = row["winner"]
                won = False
            games.append((row["START DATE"], opponent, won))

        return games

    def _two_team_tiebreaker(
        self, teams: list[str], h2h: dict, data: pd.DataFrame
    ) -> list[str]:
        """
        Break a tie between two teams using MLB tiebreaker rules.

        Order of tiebreakers:
        1. Head-to-head record
        2. Intradivision record
        3. Intraleague record
        4. Last half of intraleague games
        5. Last half plus one (iterate backwards through intraleague games)

        Parameters
        ----------
        teams : list[str]
            List of two tied team abbreviations
        h2h : dict
            Head-to-head record dict with (winner, loser): wins format
        data : pd.DataFrame
            DataFrame with season game results

        Returns
        -------
        list[str]
            Teams sorted from best to worst (winner first)
        """
        team1, team2 = teams[0], teams[1]

        # 1. Head-to-head
        t1_h2h_wins = h2h.get((team1, team2), 0)
        t2_h2h_wins = h2h.get((team2, team1), 0)
        if t1_h2h_wins > t2_h2h_wins:
            return [team1, team2]
        if t2_h2h_wins > t1_h2h_wins:
            return [team2, team1]

        # 2. Intradivision record
        t1_intradiv = self._get_intradiv_record(team1, data)
        t2_intradiv = self._get_intradiv_record(team2, data)
        if t1_intradiv["pct"] > t2_intradiv["pct"]:
            return [team1, team2]
        if t2_intradiv["pct"] > t1_intradiv["pct"]:
            return [team2, team1]

        # 3. Intraleague record
        t1_intraleague = self._get_intraleague_record(team1, data)
        t2_intraleague = self._get_intraleague_record(team2, data)
        if t1_intraleague["pct"] > t2_intraleague["pct"]:
            return [team1, team2]
        if t2_intraleague["pct"] > t1_intraleague["pct"]:
            return [team2, team1]

        # 4. Last half of intraleague games
        t1_last_half = self._get_last_half_intraleague_record(team1, data)
        t2_last_half = self._get_last_half_intraleague_record(team2, data)
        if t1_last_half["pct"] > t2_last_half["pct"]:
            return [team1, team2]
        if t2_last_half["pct"] > t1_last_half["pct"]:
            return [team2, team1]

        # 5. Last half plus one - iterate backwards through intraleague games
        return self._plus_one_tiebreaker([team1, team2], data)

    def _three_team_tiebreaker(
        self, teams: list[str], h2h: dict, data: pd.DataFrame
    ) -> list[str]:
        """
        Break a tie between three teams using MLB tiebreaker rules.

        If all three have identical h2h records against each other:
            a. Intradivision winning percentage
            b. Intraleague winning percentage
            c. Last half intraleague winning percentage
            d. Plus one tiebreaker

        If h2h records are NOT identical:
            a. If one team beat both others, that team wins
            b. Otherwise, rank by combined h2h winning percentage, then
               use two-team tiebreaker rules if needed

        Parameters
        ----------
        teams : list[str]
            List of three tied team abbreviations
        h2h : dict
            Head-to-head record dict
        data : pd.DataFrame
            DataFrame with season game results

        Returns
        -------
        list[str]
            Teams sorted from best to worst
        """
        team1, team2, team3 = teams[0], teams[1], teams[2]

        # Calculate h2h records among the three teams
        # For each team, count wins and losses against the other two
        h2h_records = {}
        for team in teams:
            others = [t for t in teams if t != team]
            wins = sum(h2h.get((team, other), 0) for other in others)
            losses = sum(h2h.get((other, team), 0) for other in others)
            total = wins + losses
            pct = wins / total if total > 0 else 0.0
            h2h_records[team] = {"wins": wins, "losses": losses, "pct": pct}

        # Check if all three have identical h2h records
        pcts = [h2h_records[t]["pct"] for t in teams]
        all_identical = len(set(pcts)) == 1

        if all_identical:
            # Path A: Use intradiv -> intraleague -> last half -> plus one
            return self._three_team_tiebreaker_identical_h2h(teams, data)
        else:
            # Path B: Check if one team beat both others
            for team in teams:
                others = [t for t in teams if t != team]
                beat_both = all(
                    h2h.get((team, other), 0) > h2h.get((other, team), 0)
                    for other in others
                )
                if beat_both:
                    # This team wins, now rank the other two
                    remaining = [t for t in teams if t != team]
                    remaining_sorted = self._two_team_tiebreaker(remaining, h2h, data)
                    return [team] + remaining_sorted

            # No team beat both others - rank by combined h2h winning percentage
            # against each other
            sorted_by_h2h = sorted(
                teams, key=lambda t: h2h_records[t]["pct"], reverse=True
            )

            # Check for ties in h2h pct
            if (
                h2h_records[sorted_by_h2h[0]]["pct"]
                > h2h_records[sorted_by_h2h[1]]["pct"]
            ):
                # First place is clear
                winner = sorted_by_h2h[0]
                remaining = [t for t in teams if t != winner]
                # Check if remaining two are tied
                if h2h_records[remaining[0]]["pct"] == h2h_records[remaining[1]]["pct"]:
                    remaining_sorted = self._two_team_tiebreaker(remaining, h2h, data)
                else:
                    remaining_sorted = sorted(
                        remaining, key=lambda t: h2h_records[t]["pct"], reverse=True
                    )
                return [winner] + remaining_sorted
            else:
                # Top two (or all three) are tied on h2h pct
                # If all three are tied, use identical h2h rules
                if (
                    h2h_records[sorted_by_h2h[1]]["pct"]
                    == h2h_records[sorted_by_h2h[2]]["pct"]
                ):
                    return self._three_team_tiebreaker_identical_h2h(teams, data)
                else:
                    # Top two are tied, third is clear
                    top_two = sorted_by_h2h[:2]
                    third = sorted_by_h2h[2]
                    top_two_sorted = self._two_team_tiebreaker(top_two, h2h, data)
                    return top_two_sorted + [third]

    def _three_team_tiebreaker_identical_h2h(
        self, teams: list[str], data: pd.DataFrame
    ) -> list[str]:
        """
        Break a three-team tie when all teams have identical h2h records.

        Order:
        a. Intradivision winning percentage
        b. Intraleague winning percentage
        c. Last half intraleague winning percentage
        d. Plus one tiebreaker

        Parameters
        ----------
        teams : list[str]
            List of three tied team abbreviations
        data : pd.DataFrame
            DataFrame with season game results

        Returns
        -------
        list[str]
            Teams sorted from best to worst
        """
        # a. Intradivision
        intradiv = {t: self._get_intradiv_record(t, data)["pct"] for t in teams}
        sorted_teams = sorted(teams, key=lambda t: intradiv[t], reverse=True)
        if (
            intradiv[sorted_teams[0]]
            > intradiv[sorted_teams[1]]
            > intradiv[sorted_teams[2]]
        ):
            return sorted_teams
        # Check for partial ties
        if intradiv[sorted_teams[0]] > intradiv[sorted_teams[1]]:
            # First is clear, check remaining two
            remaining = sorted_teams[1:]
            if intradiv[remaining[0]] == intradiv[remaining[1]]:
                # Need to continue tiebreaker for remaining two
                remaining_sorted = self._two_team_tiebreaker_from_intraleague(
                    remaining, data
                )
                return [sorted_teams[0]] + remaining_sorted
            return sorted_teams
        if (
            intradiv[sorted_teams[0]]
            == intradiv[sorted_teams[1]]
            > intradiv[sorted_teams[2]]
        ):
            # Top two tied, third is clear
            top_two_sorted = self._two_team_tiebreaker_from_intraleague(
                sorted_teams[:2], data
            )
            return top_two_sorted + [sorted_teams[2]]

        # All three still tied on intradiv, move to intraleague
        # b. Intraleague
        intraleague = {t: self._get_intraleague_record(t, data)["pct"] for t in teams}
        sorted_teams = sorted(teams, key=lambda t: intraleague[t], reverse=True)
        if (
            intraleague[sorted_teams[0]]
            > intraleague[sorted_teams[1]]
            > intraleague[sorted_teams[2]]
        ):
            return sorted_teams
        if intraleague[sorted_teams[0]] > intraleague[sorted_teams[1]]:
            remaining = sorted_teams[1:]
            if intraleague[remaining[0]] == intraleague[remaining[1]]:
                remaining_sorted = self._two_team_tiebreaker_from_last_half(
                    remaining, data
                )
                return [sorted_teams[0]] + remaining_sorted
            return sorted_teams
        if (
            intraleague[sorted_teams[0]]
            == intraleague[sorted_teams[1]]
            > intraleague[sorted_teams[2]]
        ):
            top_two_sorted = self._two_team_tiebreaker_from_last_half(
                sorted_teams[:2], data
            )
            return top_two_sorted + [sorted_teams[2]]

        # c. Last half intraleague
        last_half = {
            t: self._get_last_half_intraleague_record(t, data)["pct"] for t in teams
        }
        sorted_teams = sorted(teams, key=lambda t: last_half[t], reverse=True)
        if (
            last_half[sorted_teams[0]]
            > last_half[sorted_teams[1]]
            > last_half[sorted_teams[2]]
        ):
            return sorted_teams
        if last_half[sorted_teams[0]] > last_half[sorted_teams[1]]:
            remaining = sorted_teams[1:]
            if last_half[remaining[0]] == last_half[remaining[1]]:
                remaining_sorted = self._plus_one_tiebreaker(remaining, data)
                return [sorted_teams[0]] + remaining_sorted
            return sorted_teams
        if (
            last_half[sorted_teams[0]]
            == last_half[sorted_teams[1]]
            > last_half[sorted_teams[2]]
        ):
            top_two_sorted = self._plus_one_tiebreaker(sorted_teams[:2], data)
            return top_two_sorted + [sorted_teams[2]]

        # d. Plus one tiebreaker for all three
        return self._plus_one_tiebreaker(teams, data)

    def _two_team_tiebreaker_from_intraleague(
        self, teams: list[str], data: pd.DataFrame
    ) -> list[str]:
        """
        Two-team tiebreaker starting from intraleague record (skipping h2h and intradiv).
        Used when those have already been checked in a multi-team tiebreaker.
        """
        team1, team2 = teams[0], teams[1]

        # Intraleague record
        t1_intraleague = self._get_intraleague_record(team1, data)
        t2_intraleague = self._get_intraleague_record(team2, data)
        if t1_intraleague["pct"] > t2_intraleague["pct"]:
            return [team1, team2]
        if t2_intraleague["pct"] > t1_intraleague["pct"]:
            return [team2, team1]

        return self._two_team_tiebreaker_from_last_half(teams, data)

    def _two_team_tiebreaker_from_last_half(
        self, teams: list[str], data: pd.DataFrame
    ) -> list[str]:
        """
        Two-team tiebreaker starting from last half intraleague (skipping earlier steps).
        """
        team1, team2 = teams[0], teams[1]

        # Last half of intraleague games
        t1_last_half = self._get_last_half_intraleague_record(team1, data)
        t2_last_half = self._get_last_half_intraleague_record(team2, data)
        if t1_last_half["pct"] > t2_last_half["pct"]:
            return [team1, team2]
        if t2_last_half["pct"] > t1_last_half["pct"]:
            return [team2, team1]

        # Plus one tiebreaker
        return self._plus_one_tiebreaker([team1, team2], data)

    def _four_plus_team_tiebreaker(
        self, teams: list[str], h2h: dict, data: pd.DataFrame
    ) -> list[str]:
        """
        Break a tie between four or more teams using MLB tiebreaker rules.

        Order:
        1. Team with better record against each of the other tied teams
        2. Highest winning percentage in games among tied teams
        3. Highest intradivision winning percentage
        4. Highest intraleague winning percentage
        5. Highest last half intraleague winning percentage
        6. Plus one tiebreaker

        Parameters
        ----------
        teams : list[str]
            List of four or more tied team abbreviations
        h2h : dict
            Head-to-head record dict
        data : pd.DataFrame
            DataFrame with season game results

        Returns
        -------
        list[str]
            Teams sorted from best to worst
        """
        # 1. Check if any team beat all others
        for team in teams:
            others = [t for t in teams if t != team]
            beat_all = all(
                h2h.get((team, other), 0) > h2h.get((other, team), 0)
                for other in others
            )
            if beat_all:
                # This team wins, recursively handle the rest
                remaining = [t for t in teams if t != team]
                remaining_sorted = self._break_tie(remaining, h2h, data)
                return [team] + remaining_sorted

        # 2. Highest winning percentage among tied teams
        h2h_pcts = {}
        for team in teams:
            others = [t for t in teams if t != team]
            wins = sum(h2h.get((team, other), 0) for other in others)
            losses = sum(h2h.get((other, team), 0) for other in others)
            total = wins + losses
            h2h_pcts[team] = wins / total if total > 0 else 0.0

        sorted_by_h2h = sorted(teams, key=lambda t: h2h_pcts[t], reverse=True)

        # Check if there's a clear winner
        if h2h_pcts[sorted_by_h2h[0]] > h2h_pcts[sorted_by_h2h[1]]:
            winner = sorted_by_h2h[0]
            remaining = [t for t in teams if t != winner]
            remaining_sorted = self._break_tie(remaining, h2h, data)
            return [winner] + remaining_sorted

        # Find teams tied for top h2h pct and handle them separately
        top_pct = h2h_pcts[sorted_by_h2h[0]]
        tied_for_top = [t for t in teams if h2h_pcts[t] == top_pct]

        if len(tied_for_top) == len(teams):
            # All teams still tied, move to intradiv
            return self._four_plus_tiebreaker_from_intradiv(teams, h2h, data)
        else:
            # Some teams separated, handle top group then rest
            top_sorted = self._break_tie(tied_for_top, h2h, data)
            remaining = [t for t in teams if t not in tied_for_top]
            remaining_sorted = self._break_tie(remaining, h2h, data)
            return top_sorted + remaining_sorted

    def _four_plus_tiebreaker_from_intradiv(
        self, teams: list[str], h2h: dict, data: pd.DataFrame
    ) -> list[str]:
        """
        Continue four+ team tiebreaker from intradivision record.
        """
        # 3. Intradivision
        intradiv = {t: self._get_intradiv_record(t, data)["pct"] for t in teams}
        sorted_teams = sorted(teams, key=lambda t: intradiv[t], reverse=True)

        if intradiv[sorted_teams[0]] > intradiv[sorted_teams[1]]:
            winner = sorted_teams[0]
            remaining = [t for t in teams if t != winner]
            remaining_sorted = self._break_tie(remaining, h2h, data)
            return [winner] + remaining_sorted

        # Find tied group
        top_pct = intradiv[sorted_teams[0]]
        tied_for_top = [t for t in teams if intradiv[t] == top_pct]

        if len(tied_for_top) == len(teams):
            return self._four_plus_tiebreaker_from_intraleague(teams, h2h, data)
        else:
            top_sorted = self._break_tie(tied_for_top, h2h, data)
            remaining = [t for t in teams if t not in tied_for_top]
            remaining_sorted = self._break_tie(remaining, h2h, data)
            return top_sorted + remaining_sorted

    def _four_plus_tiebreaker_from_intraleague(
        self, teams: list[str], h2h: dict, data: pd.DataFrame
    ) -> list[str]:
        """
        Continue four+ team tiebreaker from intraleague record.
        """
        # 4. Intraleague
        intraleague = {t: self._get_intraleague_record(t, data)["pct"] for t in teams}
        sorted_teams = sorted(teams, key=lambda t: intraleague[t], reverse=True)

        if intraleague[sorted_teams[0]] > intraleague[sorted_teams[1]]:
            winner = sorted_teams[0]
            remaining = [t for t in teams if t != winner]
            remaining_sorted = self._break_tie(remaining, h2h, data)
            return [winner] + remaining_sorted

        top_pct = intraleague[sorted_teams[0]]
        tied_for_top = [t for t in teams if intraleague[t] == top_pct]

        if len(tied_for_top) == len(teams):
            return self._four_plus_tiebreaker_from_last_half(teams, h2h, data)
        else:
            top_sorted = self._break_tie(tied_for_top, h2h, data)
            remaining = [t for t in teams if t not in tied_for_top]
            remaining_sorted = self._break_tie(remaining, h2h, data)
            return top_sorted + remaining_sorted

    def _four_plus_tiebreaker_from_last_half(
        self, teams: list[str], h2h: dict, data: pd.DataFrame
    ) -> list[str]:
        """
        Continue four+ team tiebreaker from last half intraleague.
        """
        # 5. Last half intraleague
        last_half = {
            t: self._get_last_half_intraleague_record(t, data)["pct"] for t in teams
        }
        sorted_teams = sorted(teams, key=lambda t: last_half[t], reverse=True)

        if last_half[sorted_teams[0]] > last_half[sorted_teams[1]]:
            winner = sorted_teams[0]
            remaining = [t for t in teams if t != winner]
            remaining_sorted = self._break_tie(remaining, h2h, data)
            return [winner] + remaining_sorted

        top_pct = last_half[sorted_teams[0]]
        tied_for_top = [t for t in teams if last_half[t] == top_pct]

        if len(tied_for_top) == len(teams):
            # 6. Plus one for all
            return self._plus_one_tiebreaker(teams, data)
        else:
            top_sorted = self._plus_one_tiebreaker(tied_for_top, data)
            remaining = [t for t in teams if t not in tied_for_top]
            remaining_sorted = self._break_tie(remaining, h2h, data)
            return top_sorted + remaining_sorted

    def _plus_one_tiebreaker(self, teams: list[str], data: pd.DataFrame) -> list[str]:
        """
        Plus one tiebreaker: iterate backwards through each team's intraleague games.

        For each team, look at their last first-half intraleague game (excluding games
        against other tied teams), then second-to-last, etc. until the tie is broken.

        Parameters
        ----------
        teams : list[str]
            List of tied team abbreviations
        data : pd.DataFrame
            DataFrame with season game results

        Returns
        -------
        list[str]
            Teams sorted from best to worst
        """
        # Get first half games only (before All-Star break)
        first_half_data = data[data["START DATE"] < self.all_star_break]

        # Get chronological intraleague games for each team, excluding games vs tied teams
        team_games = {
            team: self._get_intraleague_games_chronological(
                team, first_half_data, exclude_teams=teams
            )
            for team in teams
        }

        # Find the maximum number of games any team has
        max_games = max(len(games) for games in team_games.values())

        # Iterate backwards through games
        for i in range(1, max_games + 1):
            # For each team, get result of their i-th game from the end
            results = {}
            for team in teams:
                games = team_games[team]
                if len(games) >= i:
                    # Get i-th game from the end
                    _, _, won = games[-i]
                    results[team] = 1 if won else 0
                else:
                    # Team doesn't have this many games, treat as loss
                    results[team] = 0

            # Check if this breaks the tie
            sorted_teams = sorted(teams, key=lambda t: results[t], reverse=True)
            if results[sorted_teams[0]] > results[sorted_teams[1]]:
                # At least first place is determined
                winner = sorted_teams[0]
                remaining = [t for t in teams if t != winner]
                if len(remaining) == 1:
                    return [winner] + remaining
                # Continue tiebreaker for remaining teams
                remaining_sorted = self._plus_one_tiebreaker(remaining, data)
                return [winner] + remaining_sorted

        # If we've exhausted all games and still tied, use random selection
        return list(self.random.permutation(teams))

    def _break_tie(self, teams: list[str], h2h: dict, data: pd.DataFrame) -> list[str]:
        """
        Route to appropriate tiebreaker based on number of teams.

        Parameters
        ----------
        teams : list[str]
            List of tied team abbreviations
        h2h : dict
            Head-to-head record dict
        data : pd.DataFrame
            DataFrame with season game results

        Returns
        -------
        list[str]
            Teams sorted from best to worst
        """
        if len(teams) == 1:
            return teams
        elif len(teams) == 2:
            return self._two_team_tiebreaker(teams, h2h, data)
        elif len(teams) == 3:
            return self._three_team_tiebreaker(teams, h2h, data)
        else:
            return self._four_plus_team_tiebreaker(teams, h2h, data)

    def _select_playoff_teams(
        self,
        results: pd.DataFrame,
        h2h: dict,
        data: pd.DataFrame,
        league: str,
        n_division_winners: int = 1,
        n_wild_cards: int = 3,
    ) -> tuple[list[str], list[str]]:
        """
        Select and seed playoff teams for a single league using MLB tiebreaker rules.

        Parameters
        ----------
        results : pd.DataFrame
            Season results with 'Team', 'wins', 'division', 'league' columns
        h2h : dict
            Head-to-head record dict
        data : pd.DataFrame
            DataFrame with season game results
        league : str
            'AL' or 'NL'
        n_division_winners : int
            Number of division winners (always 1 per division = 3 total)
        n_wild_cards : int
            Number of wild card teams per league

        Returns
        -------
        tuple[list[str], list[str]]
            (division_winners_seeded, wild_card_winners_seeded)
            Both lists are ordered by seed (1st seed first)
        """
        league_results = results[results["league"] == league].copy()
        divisions = league_results["division"].unique()

        # Step 1: Find division winners
        division_winners = []
        for div in divisions:
            div_teams = league_results[league_results["division"] == div]
            max_wins = div_teams["wins"].max()
            tied_for_first = div_teams[div_teams["wins"] == max_wins]["Team"].tolist()

            if len(tied_for_first) == 1:
                division_winners.append(tied_for_first[0])
            else:
                # Tiebreaker needed
                if len(tied_for_first) == 2:
                    self.two_way_ties += 1
                elif len(tied_for_first) == 3:
                    self.three_way_ties += 1
                elif len(tied_for_first) == 4:
                    self.four_way_ties += 1
                sorted_teams = self._break_tie(tied_for_first, h2h, data)
                division_winners.append(sorted_teams[0])

        # Step 2: Seed division winners by record (with tiebreakers if needed)
        div_winner_records = {
            team: league_results[league_results["Team"] == team]["wins"].values[0]
            for team in division_winners
        }

        # Group by wins and apply tiebreakers within groups
        wins_to_teams = {}
        for team, wins in div_winner_records.items():
            if wins not in wins_to_teams:
                wins_to_teams[wins] = []
            wins_to_teams[wins].append(team)

        seeded_div_winners = []
        for wins in sorted(wins_to_teams.keys(), reverse=True):
            teams_at_wins = wins_to_teams[wins]
            if len(teams_at_wins) == 1:
                seeded_div_winners.extend(teams_at_wins)
            else:
                sorted_teams = self._break_tie(teams_at_wins, h2h, data)
                seeded_div_winners.extend(sorted_teams)

        # Step 3: Find wild card teams
        non_div_winners = league_results[~league_results["Team"].isin(division_winners)]
        non_div_winners = non_div_winners.sort_values("wins", ascending=False)

        # Group by wins to handle ties
        wild_card_winners = []
        remaining_spots = n_wild_cards

        # Get unique win totals in descending order
        win_totals = non_div_winners["wins"].unique()

        for wins in sorted(win_totals, reverse=True):
            if remaining_spots <= 0:
                break

            teams_at_wins = non_div_winners[non_div_winners["wins"] == wins][
                "Team"
            ].tolist()

            if len(teams_at_wins) <= remaining_spots:
                # All teams at this win total make the playoffs
                if len(teams_at_wins) == 1:
                    wild_card_winners.extend(teams_at_wins)
                else:
                    # Still need to seed them via tiebreakers
                    sorted_teams = self._break_tie(teams_at_wins, h2h, data)
                    wild_card_winners.extend(sorted_teams)
                remaining_spots -= len(teams_at_wins)
            else:
                # More teams than spots - need tiebreaker to determine who makes it
                sorted_teams = self._break_tie(teams_at_wins, h2h, data)
                wild_card_winners.extend(sorted_teams[:remaining_spots])
                remaining_spots = 0

        return seeded_div_winners, wild_card_winners

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
