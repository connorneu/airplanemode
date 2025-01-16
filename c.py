import pandas as pd

# Read '/home/kman/VS_Code/projects/AirplaneModeAI/work/Lottery_Mega_Millions_Winning_Numbers__Beginning_2002.csv' into a pandas DataFrame named `data`
def read_data(file_name):
    try:
        data = pd.read_csv(file_name)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return None

# Function to find the most common individual number in Winning Numbers
def most_common_mega_ball(data):
    mega_ball_counts = data['Mega Ball'].value_counts()
    print('a')
    print(mega_ball_counts)
    most_common_ball = mega_ball_counts.idxmax()
    print('b')
    print(most_common_ball)
    count = mega_ball_counts.max()
    print('c')
    print(count)
    
    output_message = f"The most common Mega Ball is: {most_common_ball}\n"
    output_message += f"It has occurred {count} times.\n"
    
    return pd.DataFrame([output_message], columns=['Most Common Mega Ball'])

# Main function
def main():
    file_name = '/home/kman/VS_Code/projects/AirplaneModeAI/work/Lottery_Mega_Millions_Winning_Numbers__Beginning_2002.csv'
    data = read_data(file_name)
    
    if data is not None:
        print(data.head()) # For debugging purposes
        
        output_df = most_common_mega_ball(data)
        output_df.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)

main()
