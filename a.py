import pandas as pd

def load_data(file_path):
    """Load data from CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        ui.ai_response(f"Error loading data: {e}")
        return None

def main():
    # Load data
    file_path = '/home/kman/VS_Code/projects/AirplaneModeAI/work/1000_rows_ev.csv'
    data = load_data(file_path)

    if data is not None:
        # Filter rows where 'Legislative District' is provided and 'Fuel Type' is 'Battery Electric Vehicle (BEV)'
        ev_be_data = data[(data['Legislative District'] != '') & (data['Fuel Type'] == 'Battery Electric Vehicle (BEV)')]

        # Get the Legislative District with the most number of electric vehicles
        most_ev_district = ev_be_data['Legislative District'].value_counts().idxmax()

        # Print result
        ui.ai_response(f"The {most_ev_district} Legislative District has the most number of electric vehicles.")

    else:
        ui.ai_response("No data loaded")

if __name__ == "__main__":
    main()
