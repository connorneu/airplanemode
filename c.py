import pandas as pd
import os

def read_data(file_name):
    """
    Reads data from the given file.

    Args:
        file_name (str): The name of the file to be read.

    Returns:
        DataFrame: A pandas DataFrame containing the data.
    """
    if not os.path.exists(file_name):
        raise ValueError(f"File '{file_name}' not found.")

    try:
        data = pd.read_csv(file_name, parse_dates=['/home/kman/VS_Code/projects/AirplaneModeAI/work/SB2023_EU1000.xlsx'])
        return data
    except pd.errors.EmptyDataError:
        #ui.ai_response(f"The file {file_name} is empty.")
        return None
    except Exception as e:
        #ui.ai_response(f"An error occurred while reading the file: {e}")
        raise ValueError(f"Failed to read file '{file_name}': {str(e)}")

def calculate_net_sales_sum(data):
    """
    Calculates the sum of net sales for each country.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.

    Returns:
        DataFrame: A pandas DataFrame with country names and their corresponding Net sales sums.
    """
    if 'Country' not in data.columns or 'Net sales (€ million)' not in data.columns:
        raise ValueError("The input file must contain 'Country' and 'Net sales (€ million)' columns.")

    return data.groupby('Country')['Net sales (€ million)'].sum().reset_index()

def save_data(data, file_name):
    """
    Saves the given data to a file.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.
        file_name (str): The name of the file to be saved.

    Returns:
        None
    """
    try:
        data.to_csv(file_name, index=False)
    except Exception as e:
        #ui.ai_response(f"An error occurred while saving the file: {e}")
        raise ValueError(f"Failed to save file '{file_name}': {str(e)}")

def main():
    """
    The main function.
    """
    # Corrected path
    file_name = os.path.join(os.getcwd(), 'work', 'input_file.csv')

    try:
        data = read_data(file_name)
        print(data)
        if data is not None:
            print('a    ')
            country_net_sales_sum = calculate_net_sales_sum(data)
            save_data(country_net_sales_sum, 'output.csv')
            #ui.ai_response("Data saved successfully.")
    except Exception as e:
        #ui.ai_response(f"An error occurred: {e}")
        pass
        print(e)
main()
