import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def categorize_difficulty(difficulty):
    """Categorize the difficulty level of a match based on the given difficulty rating."""
    if difficulty in [1, 2]:
        return 'Low Difficulty'
    elif difficulty == 3:
        return 'Medium Difficulty'
    else:
        return 'High Difficulty'

def generate_difficulty_feature(df):
    """Generate a feature representing a player's average points against opponents of similar difficulty levels."""
    df['difficulty_category'] = df['difficulty'].apply(categorize_difficulty)
    df = df.sort_values(by=['name', 'season', 'gameweek'])
    avg_against_diff = df.groupby(['name', 'difficulty_category'])['total_points'].mean().reset_index(name='avg_points_against_difficulty')
    return pd.merge(df, avg_against_diff, on=['name', 'difficulty_category'], how='left')


def get_avg_ppg(df):
    """Generate a feature showcasing player average point per game for the season upto current gameweek"""
    df['total_points_upto_gw'] = df.groupby(['name', 'season'])['total_points'].apply(pd.Series.cumsum)
    df['total_points_upto_gw'] = df['total_points_upto_gw'] - df['total_points']
    df['avg_points_upto_gw'] = df.apply(lambda row: row['total_points_upto_gw'] / (row['gameweek'] - 1) 
                                        if row['gameweek'] > 1 else row['total_points'], axis=1)
    df.drop(columns=['total_points_upto_gw'], inplace=True)
    return df

def get_rolling_avg_mins(df, windows=[1, 2, 3, 4, 5]):
    """Generate avg minutes in previous x games feature"""
    df = df.sort_values(by=['name', 'season', 'gameweek'])
    for window in windows:
        df[f'avg_minutes_last_{window}'] = df.groupby('name')['minutes'].apply(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    return df


def generate_rolling_form(df, form_over=[3,5,10]):
    """Generate a recent form metric for players based on their performance in the last `n` gameweeks."""
    df = df.sort_values(by=['name', 'season', 'gameweek'])
    # Calculate the rolling average points over the last 3 matches for each player, excluding the current gameweek
    for window in form_over:
        df[f'avg_points_last_{window}'] = df.groupby('name')['total_points'].apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    return df

def generate_rolling_ict(df, features=['influence', 'creativity', 'threat'], window=3):
    """
    Generate average influence, threat, and creativity over the last x games.
    """
    df = df.sort_values(by=['name', 'season', 'gameweek'])
    for col in features:
        # Ensure no season overlap
        df[col + f'_avg_last_{window}_games'] = df.groupby(['name', 'season'])[col].apply(
            lambda group: group.rolling(window=window, min_periods=1).mean().shift()
        )

    df.fillna(0, inplace=True)    
    return df

def encode_positions(df):
    return pd.get_dummies(df, columns=['position'])

def encode_teams(df):
    """Change categorical variable of team and opponent team to encoding feature with full names."""
    df['team_encoded'] = df['team'].str.replace(' ', '_', regex=False)
    df['opponent_team_encoded'] = df['opponent_team_name'].str.replace(' ', '_', regex=False)
    team_dummies = pd.get_dummies(df['team_encoded']).add_prefix('team_')
    opponent_team_dummies = pd.get_dummies(df['opponent_team_encoded']).add_prefix('opponent_team_')
    df = df.drop(['team', 'opponent_team_name', 'team_encoded', 'opponent_team_encoded'], axis=1)
    df = pd.concat([df, team_dummies, opponent_team_dummies], axis=1)
    return df