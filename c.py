import pandas as pd

# Read '/home/kman/VS_Code/projects/AirplaneModeAI/work/Soccer players.csv' into a pandas DataFrame named `data`
data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/Soccer players.csv')

# Filter the data to include only players with stamina information
stamina_players = data[data['Stamina'] != 0]

# Group by 'ID' and find the player with the most stamina for each ID
max_stamina_player = stamina_players.loc[stamina_players.groupby('ID')['Stamina'].idxmax()]

# Print the player with the most stamina for each ID
print(max_stamina_player)

# Save the final output DataFrame as '/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv'
max_stamina_player.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
