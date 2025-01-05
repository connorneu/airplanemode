import pandas as pd

def analyze_data():
    data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/Soccer players.csv')

    # Drop rows where 'GK_DIVING' is True (not False or None)
    data.drop(data[data['GK_DIVING'] == True].index, inplace=True)

    data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)

analyze_data()
