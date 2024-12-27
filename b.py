import pandas as pd

def read_data(file_name):
    try:
        data = pd.read_csv(file_name)
        return data
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
        raise Exception("File not found")
    except pd.errors.EmptyDataError:
        print(f"No data in the file {file_name}.")
        raise Exception("No data in file")
    except pd.errors.ParserError as e:
        print(f"An error occurred while parsing {file_name}: {e}")
        raise Exception("Error parsing file")

def filter_data(data):
    print('bOEEEEE')
    required_columns = ['Status', 'Country']
    for column in required_columns:
        if column not in data.columns:
            raise Exception(f"The column '{column}' does not exist in the DataFrame.")
    
    filtered_data = data[(data['Status'] == 'active') & 
                         (data['Country'].isin(['Japan', 'Mexico', 'Hungary']))]
    
    return filtered_data

def main():
    data = read_data('/home/kman/VS_Code/projects/AirplaneModeAI/work/labels_roster.csv')
    
    try:
        filtered_data = filter_data(data)
        
        # We do not need the `to_csv` operation for our task,
        # so we simply print a message to indicate analysis completion.
        print("Analysis completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

main()
