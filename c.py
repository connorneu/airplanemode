import pandas as pd

try:
    data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/credit.csv')
except FileNotFoundError:
    pass
except pd.errors.EmptyDataError:
    raise ValueError(
        "The file '/home/kman/VS_Code/projects/AirplaneModeAI/work/credit.csv' is empty."
    )
except pd.errors.ParserError:
    raise ValueError(
        "An error occurred while parsing the file '/home/kman/VS_Code/projects/AirplaneModeAI/work/credit.csv'."
    )

# Ensure columns exist in DataFrame
required_columns = ["employment_duration", "age"]
if not all(column in data.columns for column in required_columns):
    raise ValueError(
        f"The following columns were not found in the DataFrame: {', '.join([column for column in required_columns if column not in data.columns])}"
    )

# Convert 'employment_duration' to numeric type
data["employment_duration"] = pd.to_numeric(data["employment_duration"], errors="coerce")

# Filter rows to include only individuals with employment duration > 7 years
data_filtered = data.loc[data["employment_duration"].notnull() & (data["employment_duration"] > 7)]

if "average_age" in data.columns:
    average_age = (
        data[data["age"].notnull()].groupby("age")['age'].mean().reset_index()
    )

# Filter the DataFrame to include only the specified columns
output_columns = ["other_credit", "housing", "default"]
try:
    output_data = data[output_columns]
except KeyError as e:
    raise ValueError(
        f"The following columns were not found in the DataFrame: {e}"
    )

# Filter rows to include only specified columns
output_data_filtered = output_data[["other_credit", "housing", "default"]]

try:
    # Drop missing values before writing to CSV
    output_data_filtered.dropna(inplace=True)
    output_data_filtered.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
except PermissionError:
    raise
