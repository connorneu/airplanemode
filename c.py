import pandas as pd

# Read '/home/kman/VS_Code/projects/AirplaneModeAI/work/Lottery_Mega_Millions_Winning_Numbers__Beginning_2002.csv' into a pandas DataFrame named `data`
data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/Lottery_Mega_Millions_Winning_Numbers__Beginning_2002.csv')

# Extract the Winning Numbers column and convert it to a list of numbers
winning_numbers_list = data['Winning Numbers'].tolist()

# Flatten the list of lists in the 'Winning Numbers' column
flat_list = [num for sublist in winning_numbers_list for num in sublist]

# Convert the flat list to a pandas Series, remove duplicates and sort it
winning_numbers_series = pd.Series(flat_list).value_counts().sort_index()

# Print the most frequent Power Ball number(s)
print("The most frequent Power Ball numbers are:")
print(winning_numbers_series.nlargest(1))

# Save the final output DataFrame as '/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv'
data['/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv'].value_counts().sort_values().to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=True, header=False)
