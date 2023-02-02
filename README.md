# Satchel
_An open source model for projecting MLB season outcomes_

<center><img src="https://baseballhall.org/sites/default/files/styles/fullscreen_image_popup/public/islandora_images/Paige%20Satchel%20177-79_PD.jpg?itok=-a_LfgqY" width=45%; height=auto;/></center>

## Description

Satchel is a simple MLB projection model meant to provide a basic forecast with relatively little effort. Much like the [Marcel The Monkey Forecasting System](http://tangotiger.net/marcel/), it is not the most advanced projection method and its projections shouldn't be treated as the word of God. I like to describe it as a minimum effort projection. Satchel essentially boils each game in a given season down to a weighted coin flip based on the amount of talent on each team's 40-man roster and then flips that coin thousands of times.

Team talent is calculated by summing up the total WAR on their 40-man roster given by FanGraphs' [Depth Charts projections](https://www.fangraphs.com/projections?pos=all&stats=bat&type=rfangraphsdc), then dividing that number by the mean (or median if you so chose) total for the league. By default, Satchel uses the Bradley-Terry model to calculate probabilities. With this model, the probability that team _A_ beats team _B_ is:

<center><i>P(A Wins) = exp(T<sub>A</sub>) &#247; [exp(T<sub>A</sub>) + exp(T<sub>B</sub>)]</i></center>

_T<sub>i</sub>_ is the talent level of team _i_.

Satchel also supports Elo-style probability calculations where the probability team _A_ beats team _B_ is instead given by:

<center><i>1 &#247; (1 + 10<sup>[T<sub>A</sub> - T<sub>B</sub>] / 400</sup>)</i></center>

Once each team's talent has been calculated, Satchel will simulate each season 10,000 times (the user can change that number if they wish), adding random noise to each team's talent each time. The results are then averaged to come up with the final projected win/loss totals and the probability each team win the wild card, division, World Series, etc.

## How to use Satchel

Using Satchel is straightforward. Create an instance of the model class, tell it to run, then wait. Running the model will return a new class object, `SatchelResults`, which contains the results of the simulation and a few fun methods for analyzing those results. To get started, you only need three lines of code:

```python
from satchel.model import Satchel

model = Satchel()
results = model.simulate()
```

The `Satchel` class supports a number of optional arguments to customize your model.

* `talent_measure`: `str`
    "mean" or "median". Each team's total WAR will be compared to the
    league's `talent_measure` to determine their talent value
* `transactions`: `dict`
    Dictionary containing any transactions to include in the simulation.
    The format of the dictionary should be:
    `{player_fangraphs_id: {"team": new_team, "date": effective_date}}`. The `player_fangraphs_id` variable is the ID FanGraphs has assigned to that player. The `new_team` is the three-letter abbreviation for the team the player is being traded to. The `effective_date` variable should be formatted `YYYY-MM-DD`.
* `noise`: _bool_:
    If true, random noise will be added to each team's talent measure
    during the simulation
* `seed` : _int or float_:
    seed used for random draws, by default None
* `steamer_p_wt`: _float_:
    Weight placed on steamer pitcher projections
* `zips_p_wt`: _float_:
    Weight placed on ZIPs pitcher projections
* `steamer_b_wt`: _float_:
    Weight placed on steamer batter projections
* `zips_b_wt`: _float_:
    Weight placed on ZIPs batter projections
* `schedule`: _Path, str_:
    Path to a CSV with the season schedule
* `pitcher_proj`: _Path, str_:
    Path to a CSV with pitcher WAR projections suitable for Satchel
* `batter_proj`: _Path, str_:
    Path to a CSV with batter WAR projections suitable for Satchel
* `use_current_results`: _bool_:
    If true, Satchel will simulate the season from today's date and add
    those results to each team's current record. This includes using
    both the team's records and the player's stats on the season in the
    talent ca* lculations. If false, Satchel will simulate the full
    season using the provided schedule and pre-season projections
* `war_method`: _str_:
    Method used for calculating all player's remaining WAR. If
    `only_projections` a player's final WAR will be their WAR to date
    plus their projected WAR multiplied by the fraction of the season
    remaining. If `current_pace`, it will be their current WAR plus
    their projected WAR multiplied by the remaining fraction of the
    season and their relative production rate. The latter is calculated
    by multiplying their projection by the fraction of the season already
    played and dividing their WAR to date by that number
* `cache`: _bool_:
    If true, the new scheudle generated will be cached


After running the model, the `SatchelResults` class has a number of methods and attributes for summarizing the results.

_Attributes_
* `al_central`: A Data Frame summarizing the results of the AL Central teams
* `al_east`: A Data Frame summarizing the results of the AL East teams
* `al_west`: A Data Frame summarizing the results of the AL West teams
* `base_talent`: A DataFrame containing each team's base talemt, i.e., their talent without any noise added.
* `div_counter`: Counts the number of times each team won their division in the simulations
* `full_seasons`: A list of DataFrames, each representing a simulated season.
* `league_counter`: Counts the number of times each team won their league in the simulations.
* `merged_schedule`: If the simulations are run mid-season and current results are merged in, this will be a DataFrame with the season-to-date results and remaining schedule merged together.
* `n`: The number of times the simulations were run.
* `nl_central`: A Data Frame summarizing the results of the NL Central teams
* `nl_east`: A Data Frame summarizing the results of the NL East teams
* `nl_west`: A Data Frame summarizing the results of the NL West teams
* `noise`: The noise added to each team's talent in each simulation.
* `playoff_counter`: Counts the number of times each team made the playoffs in the simulations.
* `playoff_matchups`: A DataFrame containing all the playoff matchups in each simulation.
* `results_df`: A DataFrame containing the full results of each simulation
* `schedule`: A DataFrame with the schedule used in the simulations.
* `season_summary`: A DataFrame summarizing the results.
* `seed`: The random seed used in the simulation.
* `trades`: A dictionary with all the trades used in the simulations.
* `wc_counter`: Counts the number of times each team won a wild card spot in the simulations.
* `ws_counter`: Counts the number of times each team won the World Series in the simulations.

_Methods_

* `boxplot()`: Creates a boxplot that shows the distribution of wins for each team.
* `season_to_date()`: Creates a table that shows season results to date, and the projected remainder of the season.
* `season_percentile(team, wins)`: Returns the percentile of the wins distribution for the specified team the given number of wins would be. 
* `results_dist_chart(team, cmap)`: Creates a bar chart showing the distribution of results for the specified team.
* `results_grid()`: Creates a grid of bar charts showing the distribution of results for each team.
* `results_scatter(team, offset, y)`: Creates a scatter plot that shows the distribution of results for the specified team.

## General remarks

Despite its methodological simplicity, Satchel does a pretty good job. When I compared it to FanGraphs' projections in 2021, it [held its own](https://andersonfrailey.github.io/blog/Satchel-2021-Autopsy.html). And it can be used for fun experiments like [trying to get the Angels to the postseason](https://andersonfrailey.github.io/blog/Can-We-Get-the-2022-Angels-to-the-Postseason.html) (a very hard task). Is this whole thing a bit overdone for how simple the model is? Possibly. And I don't care.