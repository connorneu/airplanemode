import pandas as pd

# Read 'input_file.csv' into a pandas DataFrame named `data`
data = pd.read_csv('input_file.csv')

# Filter the dataframe to include only rows where Make and Model are either BEV or PHEV
electric_vehicles = data[(data['Model Year'] == 1) & (data['Make'].str.contains('BEV', case=False)) | (data['Model'].str.contains('PHEV', case=False)]

# Count the number of Electric Vehicles in WA state, using the County column for filtering
electric_vehicles_in_wa = electric_vehicles[electric_vehicles['County'] == 'WA'].shape[0]

# Save the final output DataFrame as 'doData_Output.csv'
electric_vehicles_in_wa.to_csv('doData_Output.csv', index=False)
