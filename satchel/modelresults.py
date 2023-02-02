import pandas as pd
from dataclasses import dataclass
from collections import Counter
from typing import Union
from pybaseball import standings
from .schedules.createschedule import YEAR
from . import plotting
from . import constants


@dataclass
class SatchelResults:
    """Results class for the Satchel model"""

    ws_counter: Counter
    league_counter: Counter  # league championships
    div_counter: Counter  # division championships
    wc_counter: Counter  # wild card appearances
    playoff_counter: Counter  # any postseason appearance
    results_df: pd.DataFrame
    playoff_matchups: pd.DataFrame
    base_talent: pd.DataFrame
    n: int
    trades: dict
    schedule: pd.DataFrame
    merged_schedule: pd.DataFrame
    noise: dict  # the noise added to each team's talent in every season
    full_seasons: list  # full season results for each simulation
    seed: Union[int, None]  # seed used for the simulation

    def __post_init__(self):
        """Set all of the default values calculated from the results"""
        for col in self.playoff_matchups.columns:
            attrname = col.replace(" ", "_").lower()
            setattr(self, attrname, self._clean_matchup(self.playoff_matchups[col]))

        # summary stats
        mean_wins = pd.DataFrame(
            self.results_df.groupby("Team")["wins"].mean().round(0).astype(int)
        )
        mean_wins.columns = ["Mean Wins"]
        mean_loss = pd.DataFrame(
            self.results_df.groupby("Team")["losses"].mean().round(0).astype(int)
        )
        mean_loss.columns = ["Mean Losses"]
        max_wins = pd.DataFrame(self.results_df.groupby("Team")["wins"].max())
        max_wins.columns = ["Max Wins"]
        min_wins = pd.DataFrame(self.results_df.groupby("Team")["wins"].min())
        min_wins.columns = ["Min Wins"]
        stdev = pd.DataFrame(self.results_df.groupby("Team")["wins"].std())
        stdev.columns = ["St.Dev"]
        summary = pd.concat([mean_wins, mean_loss, max_wins, min_wins, stdev], axis=1)
        summary.reset_index(inplace=True)

        # calculate play off odds
        wc_winner = {
            team: round(_n / self.n * 100, 2)
            for team, _n in dict(self.wc_counter).items()
        }
        div_champs = {
            team: round(_n / self.n * 100, 2)
            for team, _n in dict(self.div_counter).items()
        }
        playoffs = {
            team: round(_n / self.n * 100, 2)
            for team, _n in dict(self.playoff_counter).items()
        }
        league_champs = {
            team: round(_n / self.n * 100, 2)
            for team, _n in dict(self.league_counter).items()
        }
        ws_champs = {
            team: round(_n / self.n * 100, 2)
            for team, _n in dict(self.ws_counter).items()
        }
        summary["Win Division (%)"] = summary["Team"].map(div_champs)
        summary["Make Wild Card (%)"] = summary["Team"].map(wc_winner)
        summary["Make Playoffs (%)"] = summary["Team"].map(playoffs)
        summary["Win League (%)"] = summary["Team"].map(league_champs)
        summary["Win WS (%)"] = summary["Team"].map(ws_champs)
        summary.fillna(0, inplace=True)
        summary["Division"] = summary["Team"].map(constants.DIVS)
        self.season_summary = summary
        self.alwest = summary[summary["Division"] == "AL West"].sort_values(
            "Mean Wins", ascending=False
        )
        self.alcentral = summary[summary["Division"] == "AL Central"].sort_values(
            "Mean Wins", ascending=False
        )
        self.aleast = summary[summary["Division"] == "AL East"].sort_values(
            "Mean Wins", ascending=False
        )
        self.nlwest = summary[summary["Division"] == "NL West"].sort_values(
            "Mean Wins", ascending=False
        )
        self.nlcentral = summary[summary["Division"] == "NL Central"].sort_values(
            "Mean Wins", ascending=False
        )
        self.nleast = summary[summary["Division"] == "NL East"].sort_values(
            "Mean Wins", ascending=False
        )

    def boxplot(self):
        """Create boxplot that shows the distribution of wins for each team"""
        return plotting.boxplot(self.results_df)

    def results_grid(self):
        """Create a grid of results distribution histograms"""
        return plotting.results_grid(self.results_df)

    def results_dist_chart(
        self, team, cmap=["darkgrey", "orange", "green", "red", "blue"]
    ):
        """Create a results distribution for the specified team

        Parameters
        ----------
        team : str
            String with the team name
        cmap : list, optional
            List of colors to use in the plot, by default
            ["darkgrey", "orange", "green", "red", "blue"]
        Returns
        -------
        figure
            Matplotlib figure with the chart
        """
        return plotting.results_dist_chart(
            self.results_df[self.results_df["Team"] == team],
            title=team,
            cmap=cmap,
        )

    def results_scatter(self, team, offset=0.05, y=0):
        plot_data = self.results_df[self.results_df["Team"] == team].copy()
        mean_wins = self.season_summary["Mean Wins"][
            self.season_summary["Team"] == team
        ].iloc[0]
        return plotting.results_scatter(
            plot_data, mean_wins=mean_wins, offset=offset, y=y
        )

    def season_percentile(self, team, wins):
        """Return the percentile of the wins distribution of the specified team
        the given number of wins would be in

        Parameters
        ----------
        team : str
            Three letter abreviation for the team
        wins : int
            Number of wins

        Returns
        -------
        float
            Percentile of the wins distribution the given number of wins is in
        """
        subresults = self.results_df[self.results_df["Team"] == team]
        return (subresults["wins"] < wins).sum() / subresults.shape[0]

    def season_to_date(self):
        """
        Create a table showing results to date and the remaining season projected

        Returns
        -------
        pd.DataFrame
            Table with record to date and final projections
        """
        midseason = pd.concat(standings(YEAR))
        midseason["Team"] = midseason["Tm"].map(constants.NAME_TO_ABBR)
        midseason["W"] = midseason["W"].astype(int)
        midseason["L"] = midseason["L"].astype(int)
        final = pd.merge(
            midseason,
            self.season_summary[
                ["Team", "Mean Wins", "Mean Losses", "Make Playoffs (%)"]
            ],
            on=["Team"],
        )
        final.rename(
            columns={
                "W": "Wins to Date",
                "L": "Losses to Date",
                "Mean Wins": "Projected Wins",
                "Mean Losses": "Projected Losses",
            },
            inplace=True,
        )
        final["Wins RoS"] = final["Projected Wins"] - final["Wins to Date"]
        final["Losses RoS"] = final["Projected Losses"] - final["Losses to Date"]
        final_cols = [
            "Team",
            "Wins to Date",
            "Losses to Date",
            "Wins RoS",
            "Losses RoS",
            "Projected Wins",
            "Projected Losses",
            "Make Playoffs (%)",
        ]

        return (
            final[final_cols]
            .sort_values("Projected Wins", ascending=False)
            .reset_index(drop=True)
        )

    ####### Private methods #######

    @staticmethod
    def _clean_matchup(series):
        """
        Return a clean DataFrame with the probability of all playoff matchups
        """
        matchups = pd.DataFrame(series.value_counts(normalize=True))
        matchups.reset_index(inplace=True)
        matchups.columns = ["Matchup", "Probability"]
        # sort to ensure equality when comparing results
        matchups.sort_values(
            ["Probability", "Matchup"], ascending=False, inplace=True, kind="mergesort"
        )
        matchups.reset_index(inplace=True, drop=True)
        return matchups

    def __str__(self):
        return f"Simulation Parameters:\n\t n:{self.n}\n\t trades: {self.trades}"

    def __eq__(self, __o: object) -> bool:
        # compare all of the objects of the results to see if they're equal
        for attr, val in self.__dict__.items():
            if isinstance(val, pd.DataFrame):
                otherval = getattr(__o, attr).reset_index(drop=True)
                if not val.reset_index(drop=True).equals(otherval):
                    return False
            elif attr == "full_seasons":
                for s1, s2 in zip(val, getattr(__o, "full_seasons")):
                    if not s1.reset_index(drop=True).equals(s2.reset_index(drop=True)):
                        return False
            else:
                if not val == getattr(__o, attr):
                    return False
        return True
