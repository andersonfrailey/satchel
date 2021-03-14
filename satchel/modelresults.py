import pandas as pd
from dataclasses import dataclass
from collections import Counter
from . import plotting
from . import constants


@dataclass
class SatchelResults:
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

    def __post_init__(self):
        """Set all of the default values calculated from the results"""
        self.nlwc_matchups = self._clean_matchup(self.playoff_matchups["NL Wild Card"])
        self.alwc_matchups = self._clean_matchup(self.playoff_matchups["AL Wild Card"])
        self.nlds_matchups = self._clean_matchup(
            pd.concat(
                [self.playoff_matchups["NLDS 1"], self.playoff_matchups["NLDS 2"]]
            )
        )
        self.alds_matchups = self._clean_matchup(
            pd.concat(
                [self.playoff_matchups["ALDS 1"], self.playoff_matchups["ALDS 2"]]
            )
        )
        self.nlcs_matchups = self._clean_matchup(self.playoff_matchups["NLCS"])
        self.alcs_matchups = self._clean_matchup(self.playoff_matchups["ALCS"])
        self.ws_matchups = self._clean_matchup(self.playoff_matchups["World Series"])

        # summary stats
        mean_wins = pd.DataFrame(self.results_df.groupby("Team")["wins"].mean())
        mean_wins.columns = ["Mean Wins"]
        mean_loss = pd.DataFrame(self.results_df.groupby("Team")["losses"].mean())
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
        return plotting.boxplot(self.results_df)

    def results_grid(self):
        return plotting.results_grid(self.results_df)

    def results_dist_chart(self, team, cmap=["red", "blue", "green", "purple"]):
        return plotting.results_dist_chart(
            self.results_df[self.results_df["Team"] == team], title=team, cmap=cmap,
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
        return matchups

    def __str__(self):
        return f"Simulation Parameters:\n\t n:{self.n}\n\t trades: {self.trades}"
