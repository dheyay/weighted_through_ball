import pandas as pd
import numpy as np
from datetime import datetime
import pytz


def get_gws():
    base_filepath = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data'
    seasons = ['2021-22', '2022-23', '2019-20', '2023-24', '2020-21']
    all_dataframes = []
    for season in seasons:
        if season != '2019-20':
            for gw in range(1,39):
                try:
                    df = pd.read_csv(base_filepath + f'/{season}/gws/gw{gw}.csv')
                    df['season'] = season
                    df['gameweek'] = gw
                    all_dataframes.append(df)
                except Exception as ex:
                    print(f"Gameweek {gw} for season {season} hasn't been played yet.")
                    break
        else:
            for gw in range(1, 48):
                try:
                    # WEBSITE HAS MAPPING ISSUE WITH GW 30 AS 39 AND SO ON TO 47 WHICH IS GW 38 (ONLY 2019-20 DATA)
                    df = pd.read_csv(base_filepath + f'/{season}/gws/gw{gw}.csv')
                    df['season'] = season
                    df['gameweek'] = gw if gw <=38 else gw-9
                    all_dataframes.append(df)
                except Exception as ex:
                    print(f"Error occurred: {ex}")
                    break
    return pd.concat(all_dataframes, ignore_index=True)
                

def add_opponent_team_info(player_data_df):
    seasons = ['2021-22', '2022-23', '2019-20', '2023-24', '2020-21']
    all_teams_df = []
    for season in seasons:
        df_team = pd.read_csv(f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/teams.csv').assign(season=season)
        df_team.rename({'id':'team', 'name':'team_name'}, axis=1,inplace=True)
        all_teams_df.append(df_team[['season', 'team', 'team_name']])
        
    df_team = pd.concat(all_teams_df)
    df_team.reset_index(inplace=True, drop=True)
    df_team.rename({'team':'team_drop'}, axis=1,inplace=True)
    player_data_df = pd.merge(player_data_df, df_team[['season', 'team_drop', 'team_name']], 
                           left_on=['season', 'opponent_team'], right_on=['season', 'team_drop'], how='left')
    player_data_df.rename({'team_name': 'opponent_team_name'}, axis=1, inplace=True)

    # Drop features not required for initial analysis -- Maybe do in another function
    drop_features = ['xP', 'expected_assists', 'expected_goals', 'expected_goals_conceded', 
                 'expected_goal_involvements', 'team_drop']
    player_data_df.drop(columns=drop_features, axis=1, inplace=True)

    return player_data_df

def get_all_fixture_df():
    player_data_dfs = []
    seasons = ['2021-22', '2022-23', '2019-20', '2023-24', '2020-21']
    for season in seasons:
        base_fixtures = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/fixtures.csv'
        base_teams = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/teams.csv'
        df_fixtures = pd.read_csv(base_fixtures).assign(season=season)
        df_teams = pd.read_csv(base_teams)

        # Add home team name and away team name to fixtures
        df_fixtures = df_fixtures.merge(
            df_teams[["code", "id", "name"]],
            left_on="team_h",
            right_on="id",
            suffixes=("", "_home"),
        ).merge(
            df_teams[["code", "id", "name"]],
            left_on="team_a",
            right_on="id",
            suffixes=("", "_away"),
        )
        player_data_dfs.append(df_fixtures)
    
    all_fixture_df = pd.concat(player_data_dfs, ignore_index=True)

    # Add fixture for each home team and away team
    return pd.concat(
        [
            all_fixture_df.rename(
                columns={
                    "team_h": "team",
                    "name": "team_name",
                    "team_h_difficulty": "difficulty",
                    "team_h_score": "score",
                    "team_a": "opponent",
                    "name_away": "opponent_name",
                    "team_a_difficulty": "opponent_difficulty",
                    "team_a_score": "opponent_score",
                }
            ).assign(was_home=True),
            all_fixture_df.rename(
                columns={
                    "team_a": "team",
                    "name_away": "team_name",
                    "team_a_difficulty": "difficulty",
                    "team_a_score": "score",
                    "team_h": "opponent",
                    "name": "opponent_name",
                    "team_h_difficulty": "opponent_difficulty",
                    "team_h_score": "opponent_score",
                }
            ).assign(was_home=False),
        ]
    ).reset_index()

def get_future_gameweeks(player_data, fixture_data, num_gameweeks=1):
    current_time = datetime.utcnow()
    current_time = current_time.replace(tzinfo=pytz.UTC)
    fixture_data['kickoff_time'] = pd.to_datetime(fixture_data['kickoff_time'])
    futr_fxtr = fixture_data[fixture_data['kickoff_time'] > current_time]

    # error handling if gameweek range does not exist
    next_gameweeks = sorted(futr_fxtr['event'].unique())[:num_gameweeks]
    nxt_gw_fxtr = futr_fxtr[futr_fxtr['event'].isin(next_gameweeks)]

    players_played_last_gw = player_data[(player_data['gameweek'] == nxt_gw_fxtr['event'].unique()[0] - 1) & (player_data['season'] == '2023-24')]

    # Duplicate players that were registered per club for the last completed gameweek
    #   for number of gameweeks we want
    fut_gw_players = []
    for gw in next_gameweeks:
        gw_x_players = players_played_last_gw.copy()
        gw_x_players['gameweek'] = gw
        fut_gw_players.append(gw_x_players)

    players_played_last_gw = pd.concat(fut_gw_players)

    # Edit last gw's data to future gameweeks's data
    drop_cols = ['difficulty', 'opponent_difficulty', 'opponent_team_name', 'opponent_team', 'kickoff_time']
    players_played_last_gw.drop(columns=drop_cols, inplace=True)
    players_played_last_gw['minutes'] = players_played_last_gw['minutes'].apply(lambda x: 60 if x <= 60 else 90)
    next_gw = players_played_last_gw.merge(nxt_gw_fxtr[['event', 'team_name', 'id', 'kickoff_time', 'opponent', 'difficulty', 'opponent_difficulty', 'opponent_name']],
                                                left_on=['team', 'gameweek'],
                                                right_on=['team_name', 'event'],
                                                how='left')
    next_gw['kickoff_time'] = pd.to_datetime(next_gw['kickoff_time'])
    next_gw['gameweek'] = next_gw['event']
    next_gw['opponent_team'] = next_gw['opponent']
    next_gw['opponent_team_name'] = next_gw['opponent_name']
    next_gw['fixture'] = next_gw['id']
    next_gw.drop(columns=['event', 'id', 'opponent', 'opponent_name', 'team_name'], inplace=True)
    cols_to_nan = ['goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'penalties_missed',
                'penalties_saved', 'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps', 
                'total_points', 'transfers_balance', 'transfers_in', 'transfers_out',
                'selected', 'own_goals', 'score', 'opponent_score']
    next_gw[cols_to_nan] = np.nan

    return next_gw

def get_all_player_data(fixtures):
    """Merges gameweek data with fixture data and future gameweek to create a master dataframe with all player data"""
    player_gameweeks = get_gws()

    all_player_data = pd.merge(player_gameweeks, 
                                 fixtures[['season', 'id', 'team_name', 'kickoff_time', 'difficulty', 'opponent_difficulty', 'score', 'opponent_score']], 
                                 left_on=['season', 'fixture', 'team'], 
                                 right_on=['season', 'id', 'team_name'], 
                                 how='left')
    
    # Convert kickoff_time to pd.datetime
    all_player_data.drop(['kickoff_time_x'], axis=1, inplace=True)
    all_player_data.rename(columns={"kickoff_time_y":"kickoff_time"}, inplace=True)
    all_player_data['kickoff_time'] = pd.to_datetime(all_player_data['kickoff_time'])
    
    # Dropping redundant columns
    all_player_data.drop(columns=['id', 'team_name'], inplace=True)
    all_player_data = add_opponent_team_info(all_player_data)
    return all_player_data
