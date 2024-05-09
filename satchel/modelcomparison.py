import matplotlib.pyplot as plt
from os import stat
from .modelresults import SatchelResults


class SatchelComparison:
    """This class can be used to compare the results of multiple Satchel
    simulations.
    """

    def __init__(self, res1: SatchelResults, res2: SatchelResults) -> None:
        """[summary]

        Parameters
        ----------
        res1 : SatchelResults
            The baseline results to compare to
        res2 : SatchelResults
            The new results, changed as a result of some transaction
        """
        if res1.seed != res2.seed:
            print(
                (
                    "WARNING: The seeds used for each model are different. "
                    "For the most accurate comparisons, use the same seed for both"
                )
            )

        # find the differences between all the teams
        self.res1 = res1
        self.res2 = res2
        for table in [
            "season_summary",
            "alwest",
            "alcentral",
            "aleast",
            "nlwest",
            "nlcentral",
            "nleast",
        ]:
            diffs = self._get_diffs(table)
            setattr(self, table, diffs)
        self.season_summary.set_index("Team", inplace=True)

    def win_dist_chart(self, team, sim1_label, sim2_label):
        # find the distribution of wins for each simulation
        wins1 = (
            self.res1.results_df["wins"][self.res1.results_df["Team"] == team]
            .value_counts(normalize=True)
            .reset_index()
            .sort_values("wins")
        )
        wins2 = (
            self.res2.results_df["wins"][self.res2.results_df["Team"] == team]
            .value_counts(normalize=True)
            .reset_index()
            .sort_values("wins")
        )
        # get changes in important variables
        wspct = self.season_summary["Win WS (%)"].loc[team]
        playoffpct = self.season_summary["Make Playoffs (%)"].loc[team]
        divpct = self.season_summary["Win Division (%)"].loc[team]
        meanwins = self.season_summary["Mean Wins"].loc[team]

        # get new values of the variables
        wspct_res2 = self.res2.season_summary["Win WS (%)"][
            self.res2.season_summary["Team"] == team
        ].values[0]
        playoffpct_res2 = self.res2.season_summary["Make Playoffs (%)"][
            self.res2.season_summary["Team"] == team
        ].values[0]
        divpct_res2 = self.res2.season_summary["Win Division (%)"][
            self.res2.season_summary["Team"] == team
        ].values[0]
        meanwins_res2 = self.res2.season_summary["Mean Wins"][
            self.res2.season_summary["Team"] == team
        ].values[0]

        # create figure
        fig, ax = plt.subplots(2)
        ax[0].plot(wins1["wins"], wins1["proportion"], color="blue", label=sim1_label)
        mean_wins1 = (wins1["wins"] * wins1["proportion"]).sum()
        ax[0].vlines(mean_wins1, 0, wins1["proportion"].max(), color="blue", alpha=0.5)
        ax[0].plot(wins2["wins"], wins2["proportion"], color="red", label=sim2_label)
        mean_wins2 = (wins2["wins"] * wins2["proportion"]).sum()
        ax[0].vlines(mean_wins2, 0, wins2["proportion"].max(), color="red", alpha=0.5)
        ax[0].legend(loc="upper left")
        ax[0].set_ylim(ymin=0)
        ax[0].set_xlabel("Wins")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("Wins Distribution")

        # text description of season changes
        items = [
            (wspct, "World Series Odds", wspct_res2),
            (playoffpct, "Playoff Odds", playoffpct_res2),
            (divpct, "Division Winner Odds", divpct_res2),
            (meanwins, "Expected Wins", meanwins_res2),
        ]
        yval = 0.8
        for var, name, res2 in items:
            ax[1].text(0.0, yval, name, size=15)
            color = "green"
            arrow = r"$\blacktriangle$"
            if var < 0:
                color = "red"
                arrow = r"$\blacktriangledown$"
            ax[1].text(0.75, yval, arrow, color=color, size=20)
            varstr = f"{abs(var):.2f}%".rjust(6, "0")
            numvar = f"{res2:.2f}%".rjust(6, "0")
            if name == "Expected Wins":
                varstr = f"{abs(var):.0f}".rjust(3, " ")
                numvar = f"{res2:.0f}".rjust(3, " ")
            ax[1].text(0.8, yval, varstr, color=color, size=15)
            ax[1].text(0.55, yval, numvar, color="black", size=15)
            yval -= 0.2

        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["left"].set_visible(False)
        ax[1].spines["bottom"].set_visible(False)
        ax[1].xaxis.set_ticks([])
        ax[1].yaxis.set_ticks([])
        fig.tight_layout()

        return fig

    ####### Private methods #######

    def _get_diffs(self, table):
        tbl1 = getattr(self.res1, table)
        tbl2 = getattr(self.res2, table)
        diff = tbl2.select_dtypes("number") - tbl1.select_dtypes("number")
        diff.insert(0, "Team", tbl1["Team"])

        return diff
