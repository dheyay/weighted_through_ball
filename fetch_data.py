import pandas as pd
import requests
import json
import time
import os
import re
import numpy as np


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
                    print(f"Error occurred: {ex}")
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
    df_team = pd.read_csv(f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/master_team_list.csv')
    df_team.rename({'team':'team_drop'}, axis=1,inplace=True)
    player_data_df = pd.merge(player_data_df, df_team[['season', 'team_drop', 'team_name']], 
                           left_on=['season', 'opponent_team'], right_on=['season', 'team_drop'], how='left')
    player_data_df.rename({'team_name': 'opponent_team_name'}, axis=1, inplace=True)

    # Drop features not required for initial analysis -- Maybe do in another function
    drop_features = ['xP', 'expected_assists', 'expected_goals', 'expected_goals_conceded', 
                 'expected_goal_involvements', 'kickoff_time', 'team_drop']
    player_data_df.drop(drop_features, inplace=True)

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


def get_future_gameweeks(player_data_df, all_fixture_df, season='2023-24'):
    # Generate next gameweek samples
    last_gameweek = player_data_df.where(player_data_df['season'] == season)['gameweek'].max()
    last_gameweek_players = player_data_df[(player_data_df['season'] == season) & (player_data_df['gameweek'] == last_gameweek)]
    filtered_data = all_fixture_df[(all_fixture_df['season'] == season) & (all_fixture_df['event'] == last_gameweek+1)]
    next_gw_players = last_gameweek_players.copy()

    # Update the rest of the information based on the next fixture for the team
    next_gw_players['gameweek'] = last_gameweek + 1
    drop_cols = ['difficulty', 'opponent_difficulty', 'opponent_team_name', 'opponent_team']
    next_gw_players.drop(columns=drop_cols, inplace=True)

    next_gw = next_gw_players.merge(filtered_data[['event', 'team_name', 'id', 'opponent', 'difficulty', 'opponent_difficulty', 'opponent_name']],
                                                left_on=['team', 'gameweek'],
                                                right_on=['team_name', 'event'],
                                                how='left')
    next_gw['gameweek'] = next_gw['event']
    next_gw['opponent_team'] = next_gw['opponent']
    next_gw['opponent_team_name'] = next_gw['opponent_name']
    next_gw['fixture'] = next_gw['id']
    next_gw.drop(columns=['event', 'id', 'opponent', 'opponent_name', 'team_name'], inplace=True)

    # Columns to set to NaN since games have not been played yet
    cols_to_nan = ['goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'penalties_missed',
                'penalties_saved', 'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps', 
                'total_points', 'minutes', 'transfers_balance', 'transfers_in', 'transfers_out',
                'selected', 'own_goals', 'score', 'opponent_score']

    next_gw[cols_to_nan] = np.nan
    player_data_df = pd.concat([player_data_df, next_gw])

    prev_gw_data = player_data_df[(player_data_df['season'] == '2023-24') & (player_data_df['gameweek'] == last_gameweek)]
    # Assign future players minutes based on their last game
    minutes_condition = prev_gw_data.set_index('name')['minutes'].apply(lambda x: 90 if x > 50 else 60).to_dict()
    player_data_df.loc[(player_data_df['season'] == '2023-24') & (player_data_df['gameweek'] == last_gameweek+1) & 
                       (player_data_df['name'].isin(minutes_condition.keys())), 'minutes'] = player_data_df['name'].map(minutes_condition)
    return player_data_df

def get_all_player_data():
    """Merges gameweek data with fixture data and future gameweek to create a master dataframe with all player data"""
    player_gameweeks = get_gws()
    fixtures = get_all_fixture_df()
    all_player_data = pd.merge(player_gameweeks, 
                                 fixtures[['season', 'id', 'team_name', 'difficulty', 'opponent_difficulty', 'score', 'opponent_score']], 
                                 left_on=['season', 'fixture', 'team'], 
                                 right_on=['season', 'id', 'team_name'], 
                                 how='left')

    # Dropping redundant columns
    all_player_data.drop(columns=['id', 'team_name'], inplace=True)
    future_gws = get_future_gameweeks(all_player_data, fixtures)

    return pd.concat([all_player_data, future_gws])

def fetch_fpl_team(team_id):
    """Fetch an FPL team's starting 11 and bench players given a team ID."""

    url = f"https://fantasy.premierleague.com/api/my-team/{team_id}/"
    response = requests.get(url)

    if response.status_code == 200:
        team_data = response.json()
        starting_11 = [player['element'] for player in team_data['picks'] if player['position'] <= 11]
        bench = [player['element'] for player in team_data['picks'] if player['position'] > 11]
        return {
            'starting_11': starting_11,
            'bench': bench
        }
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None
