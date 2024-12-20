# Import necessary libraries
import pandas as pd

# Read '/home/kman/VS_Code/projects/AirplaneModeAI/work/labels_roster.csv' into a pandas DataFrame named `data`
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

# Filter rows where status is active and country is Japan, Mexico, or Hungary
def filter_data(data):
    # Check if required columns exist in the DataFrame
    required_columns = ['Status', 'Country']
    for column in required_columns:
        if column not in data.columns:
            raise Exception(f"The column '{column}' does not exist in the DataFrame.")
    
    # Filter rows based on user's request
    filtered_data = data[(data['Status'] == 'active') & 
                         (data['Country'].isin(['Japan', 'Mexico']))]
    
    return filtered_data

# Main function to perform analysis and save output as '/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv'
def main():
    # Read '/home/kman/VS_Code/projects/AirplaneModeAI/work/labels_roster.csv' into a pandas DataFrame
    data = read_data('/home/kman/VS_Code/projects/AirplaneModeAI/work/labels_roster.csv')
    
    try:
        # Filter rows where status is active and country is Japan, Mexico, or Hungary
        filtered_data = filter_data(data)
        
        # Save the final output DataFrame as '/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv'
        filtered_data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
        
        print("Analysis completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the main function
if __name__ == "__main__":
    main()
