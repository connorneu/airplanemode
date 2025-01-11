import pandas as pd

def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error: {e}")

def subtract_days(df, column_name):
    try:
        df[column_name] -= 3
        return df
    except Exception as e:
        print(f"Error: {e}")

def main():
    file_path = 'C:/Users/beeen/Documents/Projects/airplanemode/work/4mb.xlsx'  # replace with your input csv file path
    data = read_data(file_path)

    data['last_name'] = pd.to_datetime(data['last_name'])

    doData_Output = subtract_days(data, 'mydate.3')

    try:
        doData_Output.to_csv('C:/Users/beeen/Documents/Projects/airplanemode/work/doData_Output.csv', index=False)
        print("Output saved successfully.")
    except Exception as e:
        print(f"Error: {e}")

main()