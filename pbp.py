# import all dependent packages
import requests as req
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import json
from unidecode import unidecode

# When working in Jupyter Notebook, allow all columns to be printed for 
# easier viewing of the data (Optional, but recommended), comment out for .py file
#pd.set_option('display.max_columns', None)

# ---------------------------------------------------------------------------------------------------
# Enter the desired seasons to scrape for. Example: scraping from 2022-23 through 2024-25 enter only
# the first part of the year into the variables. Season start is 2022 and season end is 2024.
# ---------------------------------------------------------------------------------------------------
seasons_start = 2022
seasons_end = 2024
seasons = [f"{year}{year+1}" for year in range(seasons_start, seasons_end + 1)]
print(seasons)

# ---------------------------------------------------------------------------------------------------

# Request all team data from the NHL API
df_teams = req.get(f'https://api.nhle.com/stats/rest/en/team')
df_teams = df_teams.json()
Teams = pd.json_normalize(df_teams, "data")

# ---------------------------------------------------------------------------------------------------

# Gather all skaters and goalies for the selected seasons.
Players = []

for triCode in Teams['triCode']:
    for season in seasons:
        
        response_roster = req.get(f'https://api-web.nhle.com/v1/roster/{triCode}/{season}')
        
        #some combinations will not exist as team was not active during that season, skip these instances
        if response_roster.status_code != 200:
            continue
        
        data_roster = response_roster.json()
        
        roster_forwards = pd.json_normalize(data_roster, 'forwards')
        roster_defensemen = pd.json_normalize(data_roster, 'defensemen')
        roster_goalies = pd.json_normalize(data_roster, 'goalies')
        
        roster_season = pd.concat(
            [roster_forwards, roster_defensemen, roster_goalies],
            ignore_index=True)
        
        roster_season['Season'] = season
        roster_season['Team'] = triCode
        
        Players.append(roster_season)
        
Players = pd.concat(Players, ignore_index=True)

Players["firstName"] = Players["firstName.default"].apply(unidecode)
Players["lastName"] = Players["lastName.default"].apply(unidecode)

Players['PlayerName'] = Players['firstName'] + " " + Players['lastName']

Player_Cols = ['id', 'PlayerName', 'sweaterNumber', 'birthCity.default',
               'birthStateProvince.default', 'birthCountry', 'Season', 'Team',
               'heightInInches', 'weightInPounds', 'positionCode', 'shootsCatches', ]

Players = Players[Player_Cols]

Players = Players.rename(columns = {
    'id':'PlayerID', 'sweaterNumber':'SweaterNumber', 'birthCity.default':'BirthCity',
    'birthStateProvince.default':'BirthState', 'birthCountry':'BirthCountry',
    'heightInInches':'HT', 'weightInPounds':'WT'
})

Cols_Convert = ['PlayerID', 'SweaterNumber', 'HT', 'WT']

Players[Cols_Convert] = Players[Cols_Convert].astype('Int64')

# ---------------------------------------------------------------------------------------------------

# Create an empty dataframe that will store the complete schedule
schedule = []

# Loop each team through the desired season
for triCode in Teams["triCode"]:
    for season in seasons:
        url_schedule = f'https://api-web.nhle.com/v1/club-schedule-season/{triCode}/{season}'
        response_schedule = req.get(url_schedule)
        
        # Some combinations will not exist as team was not active during that season, skip these instances
        if response_schedule.status_code != 200:
            continue
        
        data_schedule = response_schedule.json()
        data_schedule = pd.json_normalize(data_schedule, "games")
        schedule.append(data_schedule)

# Add data to the empty dataset and drop any duplicate rows that exist for each game
schedule = pd.concat(schedule, ignore_index=True)
schedule = schedule.drop_duplicates(subset=['id'])

# Filter out exhibition games
schedule = schedule[schedule['gameType'] > 1]

# ---------------------------------------------------------------------------------------------------

# Create an empty dataframe that will store all the pbp data
pbp = []

# Retrieve play-by-play data for the full season using each unique Game ID
for Game in schedule['id']:
    url_game = f'https://api-web.nhle.com/v1/gamecenter/{Game}/play-by-play'
    response_game = req.get(url_game)
    data_game = response_game.json()
    data_game = pd.json_normalize(data_game, "plays")
    
    data_game['game_id'] = Game
    start_year = int(str(Game)[:4])
    data_game['season'] = f'{start_year}{start_year+1}'
    
    # #Combine all pbp data into one set
    pbp.append(data_game)

pbp = pd.concat(pbp, ignore_index=True)
pbp['season'] = pbp['season'].astype(int)

# ---------------------------------------------------------------------------------------------------

# Create an empty dataframe that will store all the shift data
shifts = []

# Gather all shifts for the full season using Game ID
for Game in schedule['id']:
    url_shift = f'https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={Game}'
    response_shift = req.get(url_shift)
    data_shift = response_shift.json()
    data_shift = pd.json_normalize(data_shift, "data")
    shifts.append(data_shift)

shifts = pd.concat(shifts, ignore_index=True)

# ---------------------------------------------------------------------------------------------------
# Begin transformation of pbp data, create a copy of the original set of pbp data to 
# keep from re-scraping the pbp set as it takes the longest
# ---------------------------------------------------------------------------------------------------

pbp_transform = pbp.copy()

# Add game context using a map from the schedule data
schedule_map = schedule.set_index('id')

# Map columns from schedule data into pbp data
pbp_transform['game_type'] = pbp_transform['game_id'].map(schedule_map['gameType'])
pbp_transform['game_date'] = pbp_transform['game_id'].map(schedule_map['gameDate'])
pbp_transform['home_team'] = pbp_transform['game_id'].map(schedule_map['homeTeam.abbrev'])
pbp_transform['home_id'] = pbp_transform['game_id'].map(schedule_map['homeTeam.id'])
pbp_transform['away_team'] = pbp_transform['game_id'].map(schedule_map['awayTeam.abbrev'])
pbp_transform['away_id'] = pbp_transform['game_id'].map(schedule_map['awayTeam.id'])

# To perform any time difference of events the game clock will need to be converted to seconds and adjusted for each period
pbp_transform['game_seconds'] = pbp_transform['timeInPeriod'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
pbp_transform['game_seconds'] = np.where(pbp_transform['game_type'] != 1, (pbp_transform['periodDescriptor.number'] - 1) * 1200 + pbp_transform['game_seconds'], pbp_transform['game_seconds'])

pbp_transform['season_type'] = pbp_transform['game_type'].map({1: "PRE", 2: "REG"}).fillna("POST")
pbp_transform['clock_time'] = pbp_transform['timeRemaining']
pbp_transform['game_period'] = pbp_transform['periodDescriptor.number']
pbp_transform['event_team'] = np.where(
    pbp_transform['details.eventOwnerTeamId'] == pbp_transform['home_id'],
    pbp_transform['home_team'],
    pbp_transform['away_team']
)

# Assign shortened names for previous event types
event_type_map = {
    "faceoff": "FAC",
    "blocked-shot": "BLK",
    "hit": "HIT",
    "penalty": "PEN",
    "giveaway": "GIVE",
    "shot-on-goal": "SHOT",
    "takeaway": "TAKE",
    "stoppage": "STOP",
    "delayed-penalty": "DPEN",
    "goal": "GOAL",
    "missed-shot": "MISS",
    "period-start": "PSTR",
    "period-end": "PEND",
    "game-end": "GEND",
    "shootout-complete": "ENDSO",
    "failed-shot-attempt": "FSHOT"
}
# Map changes for event types
pbp_transform['event_type'] = pbp_transform['typeDescKey'].map(event_type_map)

# Assign new names to event details
event_detail_map = {
    "slap": "SLAP",
    "snap": "SNAP",
    "backhand": "BACKHAND",
    "wrist": "WRIST",
    "tip-in": "TIP-IN",
    "wrap-around": "WRAP-AROUND",
    "deflected": "DEFLECTED",
    "bat": "BAT",
    "poke": "POKE",
    "between-legs": "BETWEEN-LEGS",
    "cradle": "CRADLE"
}

# Map changes for event details
pbp_transform['event_detail'] = pbp_transform['details.shotType'].map(event_detail_map)
pbp_transform['penalty_type'] = pbp_transform['details.descKey']
pbp_transform['penalty_duration'] = pbp_transform['details.duration']

# Assign new labels to zone codes
zone_map = {"N": "Neu", "O": "Off", "D": "Def"}
pbp_transform["event_zone"] = pbp_transform["details.zoneCode"].map(zone_map)

# Create dummy variable for home and away goals
pbp_transform["home_goal"] = ((pbp_transform["event_type"] == "GOAL") & 
                         (pbp_transform["event_team"] == pbp_transform["home_team"])).astype(int)
pbp_transform["away_goal"] = ((pbp_transform["event_type"] == "GOAL") & 
                         (pbp_transform["event_team"] == pbp_transform["away_team"])).astype(int)

# Create new column for x and y coordinates
pbp_transform["xC"] = pbp_transform["details.xCoord"]
pbp_transform["yC"] = pbp_transform["details.yCoord"]

# Create column for Player ID of Player 1 during events
pbp_transform["event_player_1_id"] = np.select(
    [
        pbp_transform["typeDescKey"] == "goal",
        pbp_transform["typeDescKey"] == "faceoff",
        pbp_transform["typeDescKey"].isin(["blocked-shot", "shot-on-goal", "missed-shot"]),
        pbp_transform["typeDescKey"] == "hit",
        pbp_transform["typeDescKey"] == "penalty",
        pbp_transform["typeDescKey"] == "takeaway",
        pbp_transform["typeDescKey"] == "giveaway",
    ],
    [
        pbp_transform["details.scoringPlayerId"],
        pbp_transform["details.winningPlayerId"],
        pbp_transform["details.shootingPlayerId"],
        pbp_transform["details.hittingPlayerId"],
        pbp_transform["details.committedByPlayerId"],
        pbp_transform["details.playerId"],  
        pbp_transform["details.playerId"],  
    ],
    default=np.nan
)

pbp_transform["event_player_1_id"] = pbp_transform["event_player_1_id"].astype("Int64")

# Create column for Player ID of Player 2 during events
pbp_transform["event_player_2_id"] = np.select(
    [
        pbp_transform["typeDescKey"] == "goal",
        pbp_transform["typeDescKey"] == "faceoff",
        pbp_transform["typeDescKey"] == "blocked-shot",
        pbp_transform["typeDescKey"] == "hit",
        pbp_transform["typeDescKey"] == "penalty"
    ],
    [
        pbp_transform["details.assist1PlayerId"],
        pbp_transform["details.losingPlayerId"],
        pbp_transform["details.blockingPlayerId"],
        pbp_transform["details.hitteePlayerId"],
        pbp_transform["details.drawnByPlayerId"]
    ],
    default=np.nan
)

pbp_transform["event_player_2_id"] = pbp_transform["event_player_2_id"].astype("Int64")

# Create column for Player ID of Player 3 during events
pbp_transform["event_player_3_id"] = np.select(
    [
        pbp_transform["typeDescKey"] == "goal"
    ],
    [
        pbp_transform["details.assist2PlayerId"]
    ],
    default=np.nan
)

pbp_transform["event_player_3_id"] = pbp_transform["event_player_3_id"].astype("Int64")

# ---------------------------------------------------------------------------------------------------
# Begin calculations for distance and angle of shots taken. Adjustments are needed for shots that
# come from beyond center ice and from behind the goal.
# ---------------------------------------------------------------------------------------------------

xC = pbp_transform['xC']
yC = pbp_transform['yC']
abs_xC = np.abs(xC)
abs_yC = np.abs(yC)
event_team = pbp_transform["event_team"]
home_team = pbp_transform["home_team"]

# Limit shot adjustments to only fenwick shots and where x and y coordinates are not missing.
# NHL tracks shots at the location they are blocked and not at the location of the shot
valid_shot = (
    xC.notna() &
    yC.notna() &
    pbp_transform['event_type'].isin(["MISS", "SHOT", "GOAL"])
)

is_home = pbp_transform['event_team'] == pbp_transform['home_team']
defending_left = pbp_transform['homeTeamDefendingSide'] == "left"
shot_distance = (89 - abs_xC)**2 + yC**2
long_shots_distance = (89 + abs_xC)**2 + yC**2
shot_angle = np.arctan(abs_yC/(89 - abs_xC)) * (180/np.pi)
behind_net_shots_angle = np.arctan(abs_yC/(abs_xC - 89)) * (180/np.pi)
long_shots_angle = np.arctan(abs(yC)/(abs(xC) + 89)) * (180/np.pi)

# Calculate shot distance
pbp_transform['shot_distance'] = np.where(valid_shot,
                                       np.where(is_home,
                                                np.where(defending_left,
                                                         np.where(xC >= 0,
                                                                  np.sqrt(shot_distance),
                                                                  np.sqrt(long_shots_distance)),
                                                         np.where(xC <= 0,
                                                                  np.sqrt(shot_distance),
                                                                  np.sqrt(long_shots_distance))),
                                                np.where(defending_left,
                                                         np.where(xC <= 0,
                                                                  np.sqrt(shot_distance),
                                                                  np.sqrt(long_shots_distance)),
                                                         np.where(xC >= 0,
                                                                  np.sqrt(shot_distance),
                                                                  np.sqrt(long_shots_distance)))), np.nan).round(2)

# Calculate shot angle
pbp_transform['shot_angle'] = np.where(valid_shot,
                                       np.where(is_home,
                                                np.where(defending_left,
                                                         np.where(xC >= 0,
                                                                  np.where(xC <= 89,
                                                                           shot_angle,
                                                                           behind_net_shots_angle),
                                                                  long_shots_angle),
                                                         np.where(xC <= 0,
                                                                  np.where(abs_xC <= 89,
                                                                           shot_angle,
                                                                           behind_net_shots_angle),
                                                                  long_shots_angle)),
                                                np.where(defending_left,
                                                         np.where(xC <= 0,
                                                                  np.where(xC >= -89,
                                                                           shot_angle,
                                                                           behind_net_shots_angle),
                                                                  long_shots_angle),
                                                         np.where(xC >= 0,
                                                                  np.where(xC <= 89,
                                                                           shot_angle,
                                                                           behind_net_shots_angle),
                                                                  long_shots_angle))), np.nan).round(2)

# Create a column for player ID that won and lost the faceoff (Will be used to analyze faceoff play at a later time
# but could be ommitted at this time from final pbp set
pbp_transform['faceoff_winner_id'] = pbp_transform['details.winningPlayerId'].astype("Int64")
pbp_transform['faceoff_loser_id'] = pbp_transform['details.losingPlayerId'].astype("Int64")

#Reduce player database to only the unique Player ID values
Players_ID = Players.sort_values(['Season','PlayerName'])
Players_ID = Players_ID.drop_duplicates(subset = ['PlayerID'], keep = 'last')

# Join player names into the PBP data using Player ID
player_map = Players_ID.set_index('PlayerID')

# Map the columns into the pbp data
pbp_transform['event_player_1'] = pbp_transform['event_player_1_id'].map(player_map['PlayerName'])
pbp_transform['event_player_2'] = pbp_transform['event_player_2_id'].map(player_map['PlayerName'])
pbp_transform['event_player_3'] = pbp_transform['event_player_3_id'].map(player_map['PlayerName'])
pbp_transform['faceoff_winner'] = pbp_transform['details.winningPlayerId'].map(player_map['PlayerName'])
pbp_transform['faceoff_winner_hand'] = pbp_transform['details.winningPlayerId'].map(player_map['shootsCatches'])
pbp_transform['faceoff_winner_pos'] = pbp_transform['details.winningPlayerId'].map(player_map['positionCode'])
pbp_transform['faceoff_loser'] = pbp_transform['details.losingPlayerId'].map(player_map['PlayerName'])
pbp_transform['faceoff_loser_hand'] = pbp_transform['details.losingPlayerId'].map(player_map['shootsCatches'])
pbp_transform['faceoff_loser_pos'] = pbp_transform['details.losingPlayerId'].map(player_map['positionCode'])
pbp_transform['event_player_1_sweater'] = pbp_transform['event_player_1_id'].map(player_map['SweaterNumber'])
pbp_transform['event_player_2_sweater'] = pbp_transform['event_player_2_id'].map(player_map['SweaterNumber'])
pbp_transform['event_player_3_sweater'] = pbp_transform['event_player_3_id'].map(player_map['SweaterNumber'])
pbp_transform['CommittedBy'] = pbp_transform['details.committedByPlayerId'].map(player_map['PlayerName'])
pbp_transform['DrawnBy'] = pbp_transform['details.drawnByPlayerId'].map(player_map['PlayerName'])

# Group by Game ID and create running scores for each team during the game
pbp_transform["home_score"] = pbp_transform.groupby("game_id")["home_goal"].cumsum()
pbp_transform["away_score"] = pbp_transform.groupby("game_id")["away_goal"].cumsum()

# Create game state for each event of the game, always presented in context of the home team
pbp_transform["game_score_state"] = pbp_transform["home_score"].astype(str) + "v" + pbp_transform["away_score"].astype(str)

pbp_transform['event_index'] = (
    pbp_transform
    .groupby(['game_id', 'season'])
    .cumcount()
    .add(1)
)

# ---------------------------------------------------------------------------------------------------
# Begin transformation of shift data and use it to create an account of the players that are
# on the ice for each event during the course of the game.
# ---------------------------------------------------------------------------------------------------

# Add full player names to the shift data
shifts['fullName'] = shifts['playerId'].map(player_map['PlayerName'])

# Convert shift start time into seconds
shifts["startTimeSeconds"] = pd.to_timedelta(
    '00:' + shifts['startTime']
).dt.total_seconds()

# Convert the end time into seconds
shifts["endTimeSeconds"] = pd.to_timedelta(
    '00:' + shifts['endTime']
).dt.total_seconds()

# Adjust shift start time for each period
shifts["globalStartTime"] = (
    (shifts["period"] - 1) * 20 * 60
    + shifts["startTimeSeconds"]
)

# Adjust shift end time for each period
shifts["globalEndTime"] = (
    (shifts["period"] - 1) * 20 * 60
    + shifts["endTimeSeconds"]
)

# Create a dataset limited to only goaltenders
Goalies = Players_ID[Players_ID['positionCode'] == 'G']

# Filter the player data to only goaltender player Id values
goalie_id = Goalies['PlayerID'].unique().tolist()

# Filter the shift data to only goaltenders
goalie_shifts = shifts[shifts['playerId'].isin(goalie_id)]

# Assign pbp data to a dictionary
pbp_event_dict = pbp_transform.to_dict(orient='records')

# Create an empty set for pbp events
pbp_event_list = []

events_stop = ['GOAL', 'STOP', 'PEN']

for i, play in enumerate(pbp_event_dict):

    # skip last play in entire list
    if i == len(pbp_event_dict) - 1:
        continue

    next_play = pbp_event_dict[i + 1]

    game = play['game_id']
    game_period = play['game_period']

    # ------------------------------------------------------------------------------------------------
    # Prevent cross-game mixing by checking that the current row has the same Game ID as the next row
    # ------------------------------------------------------------------------------------------------
    if next_play['game_id'] != game:
        continue

    # Assign shift start and end to variables
    shift_start = shifts['globalStartTime']
    shift_end = shifts['globalEndTime']
    
    # Assign goalie shift start and end to variables
    goalie_shift_start = goalie_shifts['globalStartTime']
    goalie_shift_end = goalie_shifts['globalEndTime']
    
    # Sssign time to variable
    game_seconds = play['game_seconds']

    # Store next event time explicitly
    next_time = next_play['game_seconds']

    # -----------------------------------------------------------
    # Determine which conditions are to be used for game seconds
    # -----------------------------------------------------------
    if play['event_type'] in events_stop:
        shift = shifts[
            (shifts['gameId'] == game)
            & (shifts['period'] == game_period)
            & (shift_start < game_seconds)
            & (shift_end >= game_seconds)
            ]
        
        goalie_shift = goalie_shifts[
            (goalie_shifts['gameId'] == game)
            & (goalie_shifts['period'] == game_period)
            & (goalie_shift_start < game_seconds)
            & (goalie_shift_end >= game_seconds)
            ]
        
    else:
        if (
            play['game_seconds'] == next_play['game_seconds']
            and next_play['event_type'] in events_stop
        ):
            shift = shifts[
            (shifts['gameId'] == game)
            & (shifts['period'] == game_period)
            & (shift_start < game_seconds)
            & (shift_end >= game_seconds)
            ]
            
            goalie_shift = goalie_shifts[
            (goalie_shifts['gameId'] == game)
            & (goalie_shifts['period'] == game_period)
            & (goalie_shift_start < game_seconds)
            & (goalie_shift_end >= game_seconds)
            ]
            
        else:
            
            shift = shifts[
            (shifts['gameId'] == game)
            & (shifts['period'] == game_period)
            & (shift_start <= game_seconds)
            & (shift_end > game_seconds)
            ]
            
            goalie_shift = goalie_shifts[
            (goalie_shifts['gameId'] == game)
            & (goalie_shifts['period'] == game_period)
            & (goalie_shift_start <= game_seconds)
            & (goalie_shift_end > game_seconds)
            ]
    # --------------------------------------------------

    # Add players to home and away list
    home_players = shift[
        shift["teamAbbrev"] == play["home_team"]
    ]["fullName"].to_list()

    away_players = shift[
        shift["teamAbbrev"] == play["away_team"]
    ]["fullName"].to_list()
    
    home_rows = goalie_shift.loc[
        goalie_shift["teamAbbrev"] == play["home_team"]
    ]
    
    if not home_rows.empty:
        home_goalie = home_rows['fullName'].values[0]
    else:
        home_goalie = None

    away_rows = goalie_shift.loc[
        goalie_shift["teamAbbrev"] == play["away_team"]
    ]
    
    if not away_rows.empty:
        away_goalie = away_rows['fullName'].values[0]
    else:
        away_goalie = None

    pbp_event_list.append({
        **play,
        "home_on_ice": home_players,
        "away_on_ice": away_players,
        "home_goalie": home_goalie,
        "away_goalie": away_goalie,
    })

# Convert the list of processed pbp events into a dataframe
on_ice_df = pd.DataFrame(pbp_event_list)

# Create the on-ice columns for the home and away team
home_cols = [f'home_on_{i}' for i in range(1, 8)]
away_cols = [f'away_on_{i}' for i in range(1, 8)]


on_ice_df[home_cols] = (
    pd.DataFrame(on_ice_df['home_on_ice'].tolist())
      .reindex(columns=range(7))
)

on_ice_df[away_cols] = (
    pd.DataFrame(on_ice_df['away_on_ice'].tolist())
      .reindex(columns=range(7))
)

on_ice_df[home_cols + away_cols] = (
    on_ice_df[home_cols + away_cols].replace({None: np.nan})
)

on_ice_df['home_goalie'] = (
    on_ice_df['home_goalie'].replace({None: np.nan})
)
on_ice_df['away_goalie'] = (
    on_ice_df['away_goalie'].replace({None: np.nan})
)

cols_keep = [
    "game_id", "game_period", "game_seconds", 'event_index',
    "home_on_1", "home_on_2", "home_on_3", "home_on_4", "home_on_5", "home_on_6", "home_on_7",
    "away_on_1", "away_on_2", "away_on_3", "away_on_4", "away_on_5", "away_on_6", "away_on_7",
    'home_goalie', 'away_goalie']

# Filter data to only columns I want to keep
on_ice_df = on_ice_df[cols_keep]

full_pbp = pd.merge(pbp_transform, on_ice_df, how='outer', on = ['game_id', 'game_period', 'game_seconds', 'event_index'])

full_pbp["home_skaters"] = (
    full_pbp[home_cols]
        .notna()                                                        # ignore NaN values
        & full_pbp[home_cols].ne(full_pbp["home_goalie"], axis=0)     # not equal to goalie
).sum(axis=1)

full_pbp["away_skaters"] = (
    full_pbp[away_cols]
        .notna()                                                        # ignore NaN values
        & full_pbp[away_cols].ne(full_pbp["away_goalie"], axis=0)     # not equal to goalie
).sum(axis=1)

full_pbp["game_strength_state"] = full_pbp["home_skaters"].astype(str) + "v" + full_pbp["away_skaters"].astype(str)

attacking_conditions = [
    # Defending left
    (full_pbp["homeTeamDefendingSide"] == "left") & (full_pbp["xC"] < -25),
    (full_pbp["homeTeamDefendingSide"] == "left") & (full_pbp["xC"].between(-25, 25)),
    (full_pbp["homeTeamDefendingSide"] == "left") & (full_pbp["xC"] > 25),

    # Defending right
    (full_pbp["homeTeamDefendingSide"] == "right") & (full_pbp["xC"] < -25),
    (full_pbp["homeTeamDefendingSide"] == "right") & (full_pbp["xC"].between(-25, 25)),
    (full_pbp["homeTeamDefendingSide"] == "right") & (full_pbp["xC"] > 25),
]

zone_choices = [
    "Def",  # left & xC < -25
    "Neu",  # left & between
    "Off",  # left & xC > 25
    "Off",  # right & xC < -25
    "Neu",  # right & between
    "Def",  # right & xC > 25
]

full_pbp["home_zone"] = np.select(attacking_conditions, zone_choices, default=np.NaN)

full_pbp['faceoff_index'] = (
    (full_pbp['event_type'] == "FAC")
    .groupby([full_pbp['game_id'], full_pbp['season']])
    .cumsum()
)

# Add a row for the handedness of the shooter on the ice

# When the event_type is equal to a shot map the handedness of the player into the new column
fenwick_shot = ['SHOT', 'MISS', 'GOAL']

full_pbp['shooter_hand'] = full_pbp.loc[
    (full_pbp['event_type'].isin(fenwick_shot)) 
    & (full_pbp['xC'].notna())
    & (full_pbp['yC'].notna()), 'event_player_1_id'].map(player_map['shootsCatches'])

full_pbp['shooter_pos'] = full_pbp.loc[
    (full_pbp['event_type'].isin(fenwick_shot)) 
    & (full_pbp['xC'].notna())
    & (full_pbp['yC'].notna()), 'event_player_1_id'].map(player_map['positionCode'])

# Correct positions to just differentiate between forward and defense
full_pbp['shooter_pos'] = np.select(
    [
        (full_pbp['shooter_pos'] != 'D') & (full_pbp['shooter_pos'].notna()),
        (full_pbp['shooter_pos'] == 'D') & (full_pbp['shooter_pos'].notna())
    ],
    [
        'F',
        'D'
    ],
    default=np.NaN
)

# ---------------------------------------------------------------------------------------------------
# Add description of each event using columns associated with each event type
# ---------------------------------------------------------------------------------------------------

# Add a description for the PBP event based on different things
full_pbp['opp_event_team'] = np.where(
    full_pbp['event_team'] == full_pbp['home_team'], full_pbp['away_team'], full_pbp['home_team']
)

EventType = full_pbp['event_type']
EventTeam = full_pbp['event_team']
EventZone = full_pbp['event_zone']
NotEventTeam = full_pbp['opp_event_team']
EventPlayer1 = full_pbp['event_player_1']
EventPlayer2 = full_pbp['event_player_2']
EventPlayer3 = full_pbp['event_player_3']
ShotType = full_pbp['event_detail']
Player1Sweater = full_pbp['event_player_1_sweater'].astype(str)
Player2Sweater = full_pbp['event_player_2_sweater'].astype(str)
Player3Sweater = full_pbp['event_player_3_sweater'].astype(str)
PenaltyType = full_pbp['penalty_type']
PenaltyDuration = full_pbp['penalty_duration'].astype(str)
ShotDistance = full_pbp['shot_distance'].round().astype(str)

full_pbp['Description'] = np.select(
    [
        EventType == "PSTR",
        EventType == "FAC",
        EventType == "BLK",
        EventType == "PEN",
        EventType == 'GIVE',
        EventType == 'SHOT',
        EventType == 'TAKE',
        EventType == 'DPEN',
        EventType == 'GOAL',
        EventType == 'MISS',
        EventType == 'PEND',
        EventType == 'GEND',
        EventType == 'ENDSO',
        EventType == 'FSHOT',
        EventType == 'HIT'
    ],
    [
        # Period Start
        "Period Start",
        
        # Faceoff
        (EventTeam + ' Faceoff won ' + EventZone + '. Zone - '
        + EventTeam + ' #' + Player1Sweater + ' ' + EventPlayer1
        + ' vs ' + NotEventTeam + ' #' + Player2Sweater + ' ' + EventPlayer2),
        
        # Blocked Shots
        (NotEventTeam + ' ' + EventPlayer2 + ' Shot Blocked By '
        + EventTeam + ' ' + EventPlayer1 + ', ' + EventZone),
        
        # Penalty
        (EventTeam + ' #' + Player1Sweater + ' '
        + EventPlayer1 + ' ' + PenaltyType + ' - '
        + PenaltyDuration + ' min, ' + EventZone + '. Zone Drawn By '
        + NotEventTeam + ' #' + Player2Sweater + ' ' + EventPlayer2),
        
        # iveaway
        (EventTeam + ' Giveaway - #' + Player1Sweater + ' '
        + EventPlayer1 + ', ' + EventZone + '. Zone'),
        
        # Shots
        (EventTeam + ' SOG - #' + Player1Sweater +  ' '
        + EventPlayer1 + ', ' + EventZone + '. Zone, ' + ShotDistance),
        
        # Takeaway
        (EventTeam + ' Takeaway - #' + Player1Sweater + ' '
        + EventPlayer1 + ', ' + EventZone + '. Zone'),
        
        # Delayed Penalty
        'Delayed Penalty',
        
        # Goal
        (EventTeam + ' #' + Player1Sweater + ' ' + EventPlayer1 + ' '
        + ShotType + ' Shot, ' + EventZone + ', ' + ShotDistance + ' Assists:'
        + EventPlayer2 + ' ' + EventPlayer3),
        
        # Missed Shots
        (EventTeam + ' - #' + Player1Sweater +  ' ' + EventPlayer1 + ' '
        + ShotType + ', ' + EventZone + '. Zone, ' + ShotDistance),
        
        # Period End
        'Period End',
        
        # Game End
        'Game End',
        
        # End Shootout
        'Shootout Complete',
        
        # Failed Shot
        'Failed Shot',
        
        # Hits
        EventTeam + ' #' + Player1Sweater + ' ' + EventPlayer1 + ' Hit #'
        + Player2Sweater + ' ' + EventPlayer2 + ', ' + EventZone + '. Zone' 
    ],
    default=np.nan
)

#COLUMNS THAT I WANT TO KEEP
final_columns = ['season', 'game_id', 'game_date', 'season_type', 'event_index', 'game_period',
                 'game_seconds', 'clock_time', 'event_type', 'Description', 'event_detail', 'event_zone', 
                 'event_team', 'event_player_1', 'event_player_2', 'event_player_3', 'xC', 'yC',
                 'home_on_1', 'home_on_2', 'home_on_3', 'home_on_4', 'home_on_5', 'home_on_6', 'home_on_7',
                 'away_on_1', 'away_on_2', 'away_on_3', 'away_on_4', 'away_on_5', 'away_on_6', 'away_on_7', 'home_goalie',
                 'away_goalie', 'home_team', 'away_team', 'home_skaters', 'away_skaters', 'home_score', 'away_score',
                 'game_score_state', 'game_strength_state', 'home_zone', 'shot_distance', 'shot_angle', 'faceoff_index',
                 'faceoff_winner_hand', 'faceoff_winner_pos', 'faceoff_loser_hand', 'faceoff_loser_pos', 'shooter_hand', 'shooter_pos']

final_pbp = full_pbp[final_columns].copy()
final_pbp = final_pbp.sort_values(['game_id', 'game_seconds','event_index'])
