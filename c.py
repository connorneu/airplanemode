import pandas as pd

def select_rows_by_name(data, name):
    """
    Selects all rows from a DataFrame where 'Name' column matches a given string.

    Args:
        data (pd.DataFrame): Input DataFrame.
        name (str): String to search for in the 'Name' column.

    Returns:
        pd.DataFrame: A new DataFrame containing only rows with matching names.
    """
    filtered_data = data[data['Name'].str.contains(name, case=False)]

    return filtered_data

try:
    data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/Air_Quality.csv')
except FileNotFoundError:
    print("Error: Input file not found. Please check the file name and path.")
    exit(1)

name_to_search = 'Fine particles (PM 2.5)'
filtered_data = select_rows_by_name(data, name_to_search)

if not filtered_data.empty:
    filtered_data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
else:
    print("No rows found with matching names.")
