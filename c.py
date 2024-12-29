# Select all students with low stress levels and GPAs less than 3

import pandas as pd

def read_data(file_name):
    """
    Reads data from a CSV file into a DataFrame.
    
    Args:
    - file_name (str): The name of the input CSV file.
    
    Returns:
    - data (DataFrame): A DataFrame containing student data.
    """
    try:
        # Validate that the input is a string
        if not isinstance(file_name, str):
            raise ValueError("Invalid input type for file name")
        
        data = pd.read_csv(file_name)
        return data
    
    except Exception as e:
        # Log and re-raise the exception with more informative error message
        print(f"Error reading file: {e}")
        raise Exception(f"Failed to read CSV file: {e}")

def filter_students(data):
    """
    Filters students based on their stress level and GPA.
    
    Args:
    - data (DataFrame): The DataFrame containing student data.
    
    Returns:
    - filtered_data (DataFrame): A new DataFrame with only the selected students.
    """
    # Validate that required columns exist in the DataFrame
    if 'Student_ID' not in data.columns or 'GPA' not in data.columns:
        raise ValueError("Missing column(s) in the DataFrame")
    
    try:
        # Convert stress level to lowercase for correct comparison
        filtered_data = data[(data['Stress_Level'].str.lower() == 'low') & (data['GPA'] < 3)]
        
        return filtered_data
    
    except Exception as e:
        # Log and re-raise the exception with more informative error message
        print(f"Error filtering students: {e}")
        raise Exception(f"Failed to filter students based on stress level and GPA: {e}")

def save_data(data):
    """
    Saves the filtered DataFrame to a new CSV file.
    
    Args:
    - data (DataFrame): The DataFrame containing student data.
    """
    #try:
        # Validate that the output path is valid
    print(data)
    print(type(data))
    #if not isinstance(data, pd.DataFrame) or not data.empty:
    #    raise ValueError("Invalid input for saving data")
        
    data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
    
    #except Exception as e:
    #    # Log and re-raise the exception with more informative error message
    #    print(f"Error saving data: {e}")
    #    raise Exception(f"Failed to save filtered student data to CSV file: {e}")

def main():
    # Read the input file into a DataFrame
    data = read_data('/home/kman/VS_Code/projects/AirplaneModeAI/work/student_lifestyle_dataset.csv')
    
    if not data.empty:
        #try:
            # Filter students based on their stress level and GPA
        filtered_data = filter_students(data)
        
        if not filtered_data.empty:
            print('a')        
            save_data(filtered_data)
        
        #except Exception as e:
        #    print(f"An error occurred: {e}")

main()
