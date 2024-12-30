import pandas as pd

def read_input_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found")
        exit(1)
    except pd.errors.EmptyDataError:
        print("No data in file")
        exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing the file: {e}")
        exit(1)

def calculate_pilot_warned_percentage(data):
    # Calculate percentage of 'Yes' values
    pilot_warned_yes_count = (data['PilotWarned'] == 'Y').sum()
    total_count = data.shape[0]
    return ((pilot_warned_yes_count / total_count) * 100)

def save_output_to_file(output_data, file_path):
    output_data.to_csv(file_path, index=False)

def main():
    # Ensure the correct columns exist
    required_columns = ['RecordID', 'PilotWarned']
    if not all(col in data.columns for col in required_columns):
        raise Exception("Missing column(s): {}".format(", ".join([col for col in required_columns if col not in data.columns])))

    pilot_warned_percentage = calculate_pilot_warned_percentage(data)
    return round(pilot_warned_percentage, 2)

result = main()
print(result)
