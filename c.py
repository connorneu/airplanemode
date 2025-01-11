import pandas as pd

# Read 'input_file.csv' into a pandas DataFrame named `data`
try:
    data = pd.read_csv('C:/Users/beeen/Documents/Projects/airplanemode/work/4mb.xlsx')
except FileNotFoundError:
    print("The input file 'input_file.csv' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Document the user's request in a comment at the start of the code to explain what the script does
# Analyze or manipulate the DataFrame based strictly on the user's instructions: Count the number of different genders in the gender column

# Check if 'gender' column exists in data
if 'gender' not in data.columns:
    print("The 'gender' column does not exist in the input data.")
    exit()

# Get unique values from the 'gender' column and count them, excluding missing values
gender_counts = data['gender'].value_counts(dropna=True)

# Create a new DataFrame with the counts of different genders
output_data = pd.DataFrame({'Gender Counts': gender_counts})

print(output_data)

# Save the final output DataFrame as 'doData_Output.csv'
#try:
# Rename columns to avoid potential duplicate column names
output_data.columns = ['Gender', 'Count']

    # Save the CSV file without header
output_data.to_csv('C:/Users/beeen/Documents/Projects/airplanemode/work/doData_Output.csv', index=False, 
header=False)
