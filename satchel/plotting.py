import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce


def results_dataprep(data, team):
    """
    Prep data for the results histogram
    """
    data = data[data["Team"] == team]
    df = pd.DataFrame(data.groupby("wins")["season_result"].value_counts())
    df.columns = ["count"]
    # transform to a percentage
    df["count"] /= df["count"].sum()
    df.reset_index(inplace=True)
    subdfs = [
        df[["wins", "count"]][df["season_result"] == "Division Champ"].rename(
            {"count": "div"}, axis=1
        ),
        df[["wins", "count"]][df["season_result"] == "Wild Card"].rename(
            {"count": "wc"}, axis=1
        ),
        df[["wins", "count"]][df["season_result"] == "Win World Series"].rename(
            {"count": "ws"}, axis=1
        ),
        df[["wins", "count"]][df["season_result"] == "Missed Playoff"].rename(
            {"count": "mp"}, axis=1
        ),
    ]
    plot_data = reduce(
        lambda left, right: pd.merge(left, right, on="wins", how="outer"), subdfs
    )
    plot_data.fillna(0, inplace=True)
    # create bottom of bars
    plot_data["mp_b"] = 0
    plot_data["wc_b"] = plot_data[["mp"]].sum(axis=1)
    plot_data["div_b"] = plot_data[["wc", "mp"]].sum(axis=1)
    plot_data["ws_b"] = plot_data[["div", "wc", "mp"]].sum(axis=1)
    return plot_data


def results_dist_chart(data, cmap=["red", "blue", "green", "purple"], title=""):
    """
    Create a stacked bar chart to display season outcomes
    Parameters
    ----------
    data: Results data for a single team
    cmap: Color map to use
    """
    df = pd.DataFrame(data.groupby("wins")["season_result"].value_counts())
    df.columns = ["count"]
    df["count"] /= df["count"].sum()
    df.reset_index(inplace=True)
    subdfs = [
        df[["wins", "count"]][df["season_result"] == "Division Champ"].rename(
            {"count": "div"}, axis=1
        ),
        df[["wins", "count"]][df["season_result"] == "Wild Card"].rename(
            {"count": "wc"}, axis=1
        ),
        df[["wins", "count"]][df["season_result"] == "World Series"].rename(
            {"count": "ws"}, axis=1
        ),
        df[["wins", "count"]][df["season_result"] == "Missed Playoff"].rename(
            {"count": "mp"}, axis=1
        ),
    ]
    plot_data = reduce(
        lambda left, right: pd.merge(left, right, on="wins", how="outer"), subdfs
    )
    plot_data.fillna(0, inplace=True)
    # create bottom of bars
    plot_data["mp_b"] = 0
    plot_data["wc_b"] = plot_data[["mp"]].sum(axis=1)
    plot_data["div_b"] = plot_data[["wc", "mp"]].sum(axis=1)
    plot_data["ws_b"] = plot_data[["div", "wc", "mp"]].sum(axis=1)
    # make actual plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        plot_data["wins"],
        plot_data["mp"],
        label="Missed Playoffs",
        bottom=plot_data["mp_b"],
        color=cmap[0],
    )
    ax.bar(
        plot_data["wins"],
        plot_data["wc"],
        label="Wild Card",
        bottom=plot_data["wc_b"],
        color=cmap[1],
    )
    ax.bar(
        plot_data["wins"],
        plot_data["div"],
        label="Division Champ",
        bottom=plot_data["div_b"],
        color=cmap[2],
    )
    ax.bar(
        plot_data["wins"],
        plot_data["ws"],
        label="World Series",
        bottom=plot_data["ws_b"],
        color=cmap[3],
    )
    ax.legend()
    ax.set_xlabel("Wins")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    return fig, ax


def results_grid(results):
    """Create a grid showing the distribution results for all teams

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing season results for each team

    Returns
    -------
    plt.figure
        Figure with the results
    """
    # make the grid!
    cmap = ["blue", "orange", "green", "red"]
    nrows = 5
    ncols = 6
    # set teams up so each column is a division
    teams = [
        ["ARI", "CHC", "ATL", "HOU", "CHW", "BAL"],
        ["COL", "CIN", "MIA", "LAA", "CLE", "BOS"],
        ["LAD", "MIL", "NYM", "OAK", "DET", "NYY"],
        ["SDP", "PIT", "PHI", "SEA", "KCR", "TBR"],
        ["SFG", "STL", "WSN", "TEX", "MIN", "TOR"],
    ]
    divisions = ["NL West", "NL Central", "NL East", "AL West", "AL Central", "AL East"]
    # find minimum and maximum wins for axis ranges
    min_wins = results["wins"].min()
    max_wins = results["wins"].max()
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
    for i in range(nrows):
        for j in range(ncols):
            team = teams[i][j]
            plot_data = results_dataprep(results, team)
            axs[i, j].bar(
                plot_data["wins"],
                plot_data["mp"],
                label="Missed Playoffs",
                bottom=plot_data["mp_b"],
                color=cmap[0],
            )
            axs[i, j].bar(
                plot_data["wins"],
                plot_data["wc"],
                label="Make Wild Card",
                bottom=plot_data["wc_b"],
                color=cmap[1],
            )
            axs[i, j].bar(
                plot_data["wins"],
                plot_data["div"],
                label="Win Division",
                bottom=plot_data["div_b"],
                color=cmap[2],
            )
            axs[i, j].bar(
                plot_data["wins"],
                plot_data["ws"],
                label="Win World Series",
                bottom=plot_data["ws_b"],
                color=cmap[3],
            )
            axs[i, j].set_xlim(min_wins, max_wins)
            axs[i, j].set_ylim(0, 0.075)
            axs[i, j].annotate(
                team, xy=(0.80, 0.9), xycoords="axes fraction", fontsize=12
            )
            if i == 0:
                axs[i, j].set_title(divisions[j])
            if i == nrows - 1:
                axs[i, j].set_xlabel("Wins")
                axs[i, j].axes.get_xaxis().set_visible(True)
            else:
                axs[i, j].set_xticklabels([])
            if j == 0:
                axs[i, j].set_ylabel("Frequency")
            else:
                axs[i, j].set_yticklabels([])
    handles, labels = axs[i, j].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(pad=0)

    return fig


def boxplot(results):
    """Create a boxplot that shows the distribution of wins for each team

    Parameters
    ----------
    result : pd.DataFrame
        DataFrame containing the results of all the season simulations

    Returns
    -------
    Matplotlib Figure
    """
    boxdata = [
        results["wins"][results["Team"] == team]
        for team in results["Team"].sort_values().unique()
    ]
    boxfig, boxax = plt.subplots(figsize=(10, 5))
    boxax.boxplot(boxdata, labels=results["Team"].sort_values().unique())
    boxax.tick_params(axis="x", labelrotation=45)
    boxax.set_ylabel("Wins")
    boxax.set_title("Distribution of Total Wins")

    return boxfig, boxax

