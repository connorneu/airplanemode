import pandas as pd

def analyze_user_data(data):
    # Filter data based on user's instructions
    filtered_df = data[(data['checking_balance'] == '< 0 DM') | 
                       (data['checking_balance'] > '200 DM')]
    
    education_no_debt = filtered_df[filtered_df['credit_history'].str.contains('poor')] & \
                        (filtered_df['purpose'].str.contains('education')) & \
                        (filtered_df['default'] == 'no')
                        
    # Calculate percentage of respondents in the "education" category who report no debt
    if not education_no_debt.empty:
        try:
            percentage = (len(education_no_debt) / len(filtered_df)) * 100
            
            return {'percentage': round(percentage,2)}
        except Exception as e:
            raise Exception("Error analyzing user's data: " + str(e))
    else:
        return None

data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/input_file.csv')

result = analyze_user_data(data)

if result is not None:
    result_df = pd.DataFrame([result])
    result_df.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
