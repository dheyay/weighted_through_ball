import pandas as pd
import numpy as np

def generate_rolling_form_feature(df, form_range=3):
    """Generate a recent form metric for players based on their performance in the last `n` gameweeks."""
    df = df.sort_values(by=['name', 'season', 'gameweek'])
    # Calculate the rolling average points over the last 3 matches for each player, excluding the current gameweek
    df['form_last_3'] = df.groupby('name')['total_points'].apply(lambda x: x.shift(1).rolling(window=form_range, min_periods=1).mean())
    return df

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


def get_avg_ppg(df, season=None):
    """Generate a feature showcasing player average point per game for the specified season"""
    # Calculate the cumulative sum of total_points for each player for each season
    df['cumulative_points'] = df.groupby(['name', 'season'])['total_points'].cumsum()
    df['avg_ppg'] = df['cumulative_points'] / df['gameweek']
    df.drop(columns=['cumulative_points'], inplace=True)
    return df
