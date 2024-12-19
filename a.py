import pandas as pd
from datetime import datetime

# Read '/home/kman/VS_Code/projects/AirplaneModeAI/work/all_youtube_analytics.csv' into a pandas DataFrame named `data`
def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("The file was not found.")
        raise
    except pd.errors.EmptyDataError:
        print("The file is empty.")
        raise

# Function to calculate day of the week from 'day' column
def calculate_day_of_week(data):
    try:
        # Convert the 'day' column to datetime format
        data['day'] = pd.to_datetime(data['day'])
        
        # Extract the day of the week
        data['day_of_week'] = data['day'].dt.day_name()
        
        return data
    
    except KeyError:
        print("The column does not exist.")
        raise

# Function to save DataFrame as 'output_file.csv'
def save_data(data, output_path):
    try:
        # Save the DataFrame to a csv file
        data.to_csv(output_path, index=False)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# Main function to perform operations
def main():
    data = read_data('/home/kman/VS_Code/projects/AirplaneModeAI/work/all_youtube_analytics.csv')
    
    if not data.empty:
        # Calculate day of the week from 'day' column
        data = calculate_day_of_week(data)
        
        # Save the output DataFrame as '/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv'
        save_data(data, '/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv')

if __name__ == "__main__":
    main()
