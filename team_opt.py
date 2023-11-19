import requests
import pulp

def fetch_fpl_team(self):
    """Fetch an FPL team's starting 11 and bench players given a team ID."""
    url = f"https://fantasy.premierleague.com/api/my-team/{self.team_id}/"
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

def select_team(data):
    # Create a linear optimization problem
    prob = pulp.LpProblem('FPLTeamSelection', pulp.LpMaximize)

    # Create decision variables
    x = pulp.LpVariable.dicts('player_in_11', data.index, cat='Binary')
    s = pulp.LpVariable.dicts('player_as_sub', data.index, cat='Binary')

    # Objective function
    prob += pulp.lpSum(data['xP'] * x[i] for i in data.index) + pulp.lpSum(data['xP'] * s[i] for i in data.index)

    # Constraints
    prob += pulp.lpSum(x[i] for i in data.index) == 11
    prob += pulp.lpSum(s[i] for i in data.index) == 4

    # Positional constraints for starting lineup (adjust as needed)
    prob += pulp.lpSum(x[i] for i in data[data['position_GK'] == 1].index) == 1
    prob += pulp.lpSum(x[i] for i in data[data['position_DEF'] == 1].index) >= 3
    prob += pulp.lpSum(x[i] for i in data[data['position_DEF'] == 1].index) <= 5
    prob += pulp.lpSum(x[i] for i in data[data['position_MID'] == 1].index) >= 3
    prob += pulp.lpSum(x[i] for i in data[data['position_MID'] == 1].index) <= 5
    prob += pulp.lpSum(x[i] for i in data[data['position_FWD'] == 1].index) >= 1
    prob += pulp.lpSum(x[i] for i in data[data['position_FWD'] == 1].index) <= 3

    # Player can't be both in the starting lineup and a substitute
    for i in data.index:
        prob += x[i] + s[i] <= 1

    # Budget constraint
    total_budget = 100  # in millions
    prob += pulp.lpSum(data['value'] * x[i] for i in data.index) + pulp.lpSum(data['value'] * s[i] for i in data.index) <= total_budget

    # Maximum 3 players from a single team constraint
    # TODO Replace team names with actual team names when data inference is done.
    teams = ['arsenal', 'aston', 'bournemouth', 'brentford', 'brighton', 'brom', 'burnley',
         'chelsea', 'city', 'crystal', 'everton', 'forest', 'fulham', 'ham', 'leeds',
         'leicester', 'liverpool', 'luton', 'man', 'newcastle', 'norwich', 'nott', 'palace',
         'sheffield', 'southampton', 'spurs', 'utd', 'villa', 'watford', 'west', 'wolves']

    for team in teams:
        team_column = f'team_{team}'
        prob += pulp.lpSum(x[i] for i in data[data[team_column] == 1].index) + \
                pulp.lpSum(s[i] for i in data[data[team_column] == 1].index) <= 3

    # teams = data['team'].unique()
    # for team in teams:
    #     prob += pulp.lpSum(x[i] for i in data[data['team'] == team].index) + pulp.lpSum(s[i] for i in data[data['team'] == team].index) <= 3

    # Solve the problem
    prob.solve()

    # Extract selected players for starting 11 and substitutes
    starting_11 = [i for i in data.index if x[i].value() == 1]
    substitutes = [i for i in data.index if s[i].value() == 1]

    return starting_11, substitutes

