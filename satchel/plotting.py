import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from collections import defaultdict


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
        df[["wins", "count"]][df["season_result"] == "Win League"].rename(
            {"count": "cs"}, axis=1
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
    plot_data["cs_b"] = plot_data[["div", "wc", "mp"]].sum(axis=1)
    plot_data["ws_b"] = plot_data[["cs", "div", "wc", "mp"]].sum(axis=1)
    return plot_data


def results_dist_chart(
    data, cmap=["darkgrey", "orange", "green", "red", "blue"], title=""
):
    """
    Create a stacked bar chart to display season outcomes
    Parameters
    ----------
    data: Results data for a single team
    cmap: Color map to use
    """
    assert len(cmap) == 5
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
        df[["wins", "count"]][df["season_result"] == "Win World Series"].rename(
            {"count": "ws"}, axis=1
        ),
        df[["wins", "count"]][df["season_result"] == "Missed Playoff"].rename(
            {"count": "mp"}, axis=1
        ),
        df[["wins", "count"]][df["season_result"] == "Win League"].rename(
            {"count": "wl"}, axis=1
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
    plot_data["wl_b"] = plot_data[["div", "wc", "mp"]].sum(axis=1)
    plot_data["ws_b"] = plot_data[["wl", "div", "wc", "mp"]].sum(axis=1)
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
        plot_data["wl"],
        label="League Champ",
        bottom=plot_data["wl_b"],
        color=cmap[3],
    )
    ax.bar(
        plot_data["wins"],
        plot_data["ws"],
        label="World Series",
        bottom=plot_data["ws_b"],
        color=cmap[4],
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
    cmap = ["blue", "orange", "green", "darkgrey", "red"]
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
                label="Win Wild Card",
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
                plot_data["cs"],
                label="Win League",
                bottom=plot_data["cs_b"],
                color=cmap[3],
            )
            axs[i, j].bar(
                plot_data["wins"],
                plot_data["ws"],
                label="Win World Series",
                bottom=plot_data["ws_b"],
                color=cmap[4],
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
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07))
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


def results_scatter(data, mean_wins, offset=0.05, y=0):
    def assign_y(x, win_obs, y, offset):
        nobs = win_obs[x]
        yval = y + (offset * nobs)
        if nobs % 2 != 0:
            yval *= -1
        win_obs[x] += 1
        return yval

    color_map = {
        "Missed Playoffs": "darkgrey",
        "Wild Card": "orange",
        "Division Champ": "green",
        "Win League": "red",
        "Win World Series": "blue",
    }
    res_names = {
        "Missed Playoffs": "Missed Playoffs",
        "Wild Card": "Wild Card",
        "Division Champ": "Won Division",
        "Win League": "Won League",
        "Win World Series": "Won World Series",
    }
    data["res_cat"] = data["season_result"].astype("category")
    # only include values in the re-order that appear in the new order
    res_order = [
        "Missed Playoffs",
        "Wild Card",
        "Division Champ",
        "Win League",
        "Win World Series",
    ]
    new_order = [_res for _res in res_order if _res in data["season_result"].unique()]
    data["res_cat"].cat.reorder_categories(
        new_order,
        inplace=True,
    )
    data.sort_values("res_cat", inplace=True)

    win_obs = defaultdict(int)
    data["yval"] = data["wins"].apply(assign_y, win_obs=win_obs, y=y, offset=offset)
    fig, ax = plt.subplots()
    ax.axvline(x=mean_wins, color="black", label="Mean Wins")
    ax.grid(axis="x")
    for _res in color_map.keys():
        sub = data[data["season_result"] == _res]
        ax.scatter(
            sub["wins"],
            sub["yval"],
            color=color_map[_res],
            label=res_names[_res],
            alpha=0.65,
        )
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.yaxis.set_ticks([])
    ax.set_xlabel("Wins")
    ax.legend(
        bbox_to_anchor=(0.99, 1),
        title="Season Result",
        title_fontproperties={"weight": "bold"},
    )
    ax.set_ylim([data["yval"].min() - offset * 3, data["yval"].max() + offset * 3])

    return fig, ax
