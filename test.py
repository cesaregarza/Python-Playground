# %%
import math_functions, color

# %%
import pandas as pd
import numpy as np
import math_functions
from math_functions.directional_2 import DirectionalStatistics

df = pd.read_csv(f"F:\Dev\shoothouse.csv").iloc[:, 1:]
df["vx0vy0"] = np.round(df['vx'] / 5) * 5 * 1024 + np.round(df['vy'] / 5) * 5

# %%

direc_stats = []
groupby = df.groupby("vx0vy0")
for name, group in groupby:
    center = (name // 1024, name % 1024)
    ds = DirectionalStatistics(group[['ax','ay']].values, center=center)
    mean, var = ds.mean()
    ct = len(group)
    direc_stats += [[name, ct, mean, var]]

xdf = pd.DataFrame(direc_stats)
xdf.columns = ["vx0vy0", "count", "mean", "var"]
xdf.describe()

# %%
import math_functions
import pandas as pd
from math_functions.directional_2 import DirectionalStatistics
import numpy as np
xy = [
    [1, 1],
    [1, 2],
    [1, 3],
    [1, -1],
    [1, -2],
    [1, -3],
]
df = pd.DataFrame(xy, columns=["x", "y"])
ds = DirectionalStatistics.from_dataframe(df)
ds.var
# %%
ds.extrinsic_mean()
# %%
import json
import pandas as pd
#Open the json.txt file and parse it as json
with open('json.txt', 'r') as f:
    json_obj = json.load(f)

init_df = pd.json_normalize(json_obj)
init_df['player_id'] = init_df['players'].apply(lambda x: x['puuid'])
# %%
def parse_json(json_dict:dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    #Use pandas to attempt to parse the json
    initial_df = pd.json_normalize(json_dict)

    #Create an initial player dataframe which is the initial dataframe with the players column exploded
    player_df = initial_df.explode('players', ignore_index=True)[['matchInfo.matchId', 'players']]

    #Define a function to parse the player data on each individual matchId, to be used in a groupby
    def parse_player_data(player_data:pd.DataFrame) -> pd.DataFrame:
        #Create a new dataframe to hold the player data
        player_df = pd.DataFrame()

        #For each player in the match, parse their data
        for player in player_data['players']:
            #Create a new dataframe for the player
            player_df = player_df.append(pd.json_normalize(player), ignore_index=True)

        #Return the player dataframe
        return player_df
    
    #Group the player data by matchId
    player_df = player_df.groupby("matchInfo.matchId").apply(parse_player_data)

    #Rename the index columns to something more readable
    player_df.index.names = ["match_id", "position"]

    #Rename the columns to something more readable by getting only the last part of the column name
    player_df.columns = player_df.columns.str.rsplit(".", 1).str[-1]

    #Convert camel case to snake case
    player_df.columns = [
        'player_id',
        'game_name',
        'tag_line',
        'team_id',
        'party_id',
        'character_id',
        'competitive_tier',
        'player_card',
        'player_title',
        'score',
        'rounds_played',
        'kills',
        'deaths',
        'assists',
        'playtime_millis',
        'grenade_casts',
        'ability1casts',
        'ability2casts',
        'ultimate_casts'
    ]

    #Reset the index, drop the "position" column, then re-index by match_id then player_id
    player_df = player_df.reset_index().drop("position", axis=1).set_index(["match_id", "player_id"])

    #Create a new dataframe to hold the round data
    round_df = pd.json_normalize(initial_df.loc[0, 'roundResults'])

# %%
def calc_func(initial_scores:list[int], action_values:list[list[int]]) -> list[list[int]]:
    #The goal is to get 8 points for each of the 3 scores over 5 rounds with any combination of actions.

    min_score = 999
    #Generate every possible permutation of the actions of length 5
    for i in range(len(action_values)):
        for j in range(len(action_values)):
            for k in range(len(action_values)):
                for l in range(len(action_values)):
                    for m in range(len(action_values)):
                        #Generate the score for the current permutation
                        score1 = sum([action_values[x][0] for x in [i, j, k, l, m]]) + initial_scores[0]
                        score2 = sum([action_values[x][1] for x in [i, j, k, l, m]]) + initial_scores[1]
                        score3 = sum([action_values[x][2] for x in [i, j, k, l, m]]) + initial_scores[2]
                        score = (score1 - 8) ** 2 + (score2 - 8) ** 2 + (score3 - 8) ** 2
                        
                        if score < min_score:
                            min_score = score
                            min_actions = [i, j, k, l, m]

                        if score == 0:
                            return action_values[i], action_values[j], action_values[k], action_values[l], action_values[m]
    
    return [action_values[x] for x in min_actions]
# %%
