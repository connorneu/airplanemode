
import pandas as pd

# Original code with added comments and changes
data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/dataset.csv')

# Split the Player column into two separate columns for first name and last name
data[['First_Name', 'Last_Name']] = data['Player'].str.split(expand=True)

# Add a comment to indicate that this is where the user's request was documented
# The player values are preserved in the DataFrame, as requested

data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv')
