import pandas as pd

def clean_2019_data(player_data_df, fixture_data):
    """Clean the player data for the 2019-20 season by mapping missing data using the fixture data."""
    fixtures_2019_20_data = fixture_data[fixture_data['season'] == '2019-20']
    team_mapping = fixtures_2019_20_data.set_index('id').apply(lambda row: row['team_name'] if row['was_home'] else row['opponent_name'], axis=1).to_dict()
    difficulty_mapping = fixtures_2019_20_data.set_index('id')['difficulty'].to_dict()
    opponent_difficulty_mapping = fixtures_2019_20_data.set_index('id')['opponent_difficulty'].to_dict()

    # Re-defining the mask for the 2019-20 season
    mask_2019_20 = player_data_df['season'] == '2019-20'
    player_data_df.loc[mask_2019_20, 'team'] = player_data_df.loc[mask_2019_20, 'fixture'].map(team_mapping)
    player_data_df.loc[mask_2019_20, 'difficulty'] = player_data_df.loc[mask_2019_20, 'fixture'].map(difficulty_mapping)
    player_data_df.loc[mask_2019_20, 'opponent_difficulty'] = player_data_df.loc[mask_2019_20, 'fixture'].map(opponent_difficulty_mapping)

    # Adjusting the mappings for 'score' and 'opponent_score' based on the 'was_home' column in the main dataset
    score_mapping = fixtures_2019_20_data.set_index('id').apply(lambda row: row['score'] if row['was_home'] else row['opponent_score'], axis=1).to_dict()
    opponent_score_mapping = fixtures_2019_20_data.set_index('id').apply(lambda row: row['opponent_score'] if row['was_home'] else row['score'], axis=1).to_dict()
    player_data_df.loc[mask_2019_20, 'score'] = player_data_df.loc[mask_2019_20, 'fixture'].map(score_mapping)
    player_data_df.loc[mask_2019_20, 'opponent_score'] = player_data_df.loc[mask_2019_20, 'fixture'].map(opponent_score_mapping)
    return player_data_df

def fill_player_positions(player_data_df):
    """Fill missing player positions for the 2019-20 dataset"""
    player_data_df['position'].fillna('UNK', inplace=True)
    player_data_df.loc[player_data_df['position'] == 'GKP', 'position'] = 'GK'
    players_2019_20 = set(player_data_df[player_data_df['season'] == '2019-20']['name'].unique())
    players_other_seasons = set(player_data_df[player_data_df['season'] != '2019-20']['name'].unique())
    players_both_seasons = players_2019_20.intersection(players_other_seasons)
    latest_positions = player_data_df[player_data_df['name'].isin(players_both_seasons)].groupby('name')['position'].last().to_dict()
    player_data_df.loc[(player_data_df['name'].isin(players_both_seasons) & (player_data_df['season'] == '2019-20')),
                        'position'] = player_data_df['name'].map(latest_positions)
    
    # Drop players with UNK position as they are not currently playing
    player_data_df = player_data_df[player_data_df['position'] != 'UNK']
    return player_data_df

def standardize_element_ids(df):
    """Standardize element id for players using their latest element id"""
    latest_seasons = df.groupby('name')['season'].transform(max)
    latest_players_df = df[df['season'] == latest_seasons]
    name_to_latest_element = dict(zip(latest_players_df['name'], latest_players_df['element']))
    df['element'] = df['name'].map(name_to_latest_element)
    return df

def drop_cols(drop_cols, df):
    return df.drop(drop_cols, inplace=True)

def clean_player_data(all_player_df, fixtures_data):
    all_player_df = clean_2019_data(all_player_df, fixtures_data)
    all_player_df = fill_player_positions(all_player_df)
    all_player_df = standardize_element_ids(all_player_df)
    return all_player_df