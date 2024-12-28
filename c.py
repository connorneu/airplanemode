import pandas as pd

def analyze_user_data(data):
    required_columns = ['Label ID', 'Band ID', 'Name', 'Specialization', 'Status', 'Country']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Data is missing one or more required columns")

    if not (data['Status'].isin(['active', 'closed', 'unknown', 'changed name', 'on hold'])):
        raise ValueError("Invalid value in the Status column")
    if not (data['Country'].isin(data['Country'].unique())):
        raise ValueError("Duplicated country")

    filtered_data = data[(data['Status'] == 'active') & (data['Country'].isin(['Japan', 'Mexico', 'Hungary']))]
    
    return filtered_data

try:
    data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/labels_roster.csv')
except FileNotFoundError:
    print("The specified file was not found.")
    exit()

filtered_data = analyze_user_data(data)
try:
    filtered_data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
except Exception as e:
    print(f"Failed to write data to csv file: {e}")
