import pandas as pd

# Step 1: Read the input file into a DataFrame
data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/100_rows_ev.csv')

# Step 2: Remove the first letter from 'County' column
data['County'] = data['County'].str.lstrip()
print(data['County'])

# Step 3: Save the updated DataFrame to a new output file
#data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
