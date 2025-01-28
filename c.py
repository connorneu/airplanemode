import pandas as pd
import matplotlib.pyplot as plt

# Read input data into DataFrame named `data`
data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/SB2023_EU1000.xlsx')

# Ask for user's question to generate the solution
pass

if 'R&D' in user_question and 'Net Sales' in user_question:
    # Calculate ratio of R&D expenditure to net sales 
    r_d_ratio = data['R&D (€ million)'] / data['Net sales (€ million)']
    
    plt.figure(figsize=(10,8))
    plt.plot(r_d_ratio)
    plt.title(f"Ratio of Research and Development Expenditure to Net Sales ({user_question})")
    plt.xlabel("Company Index")
    plt.ylabel("Ratio")
    plt.show()
elif 'R&D' in user_question:
    # Ask the user to select company for which R&D one-year growth percentage is required
    companies = data['Company'].unique()

    while True:
        ui.ai_response("Select a Company:")
        for i, company in enumerate(companies):
            ui.ai_response(f"{i+1}. {company}")
        
pass
        company = companies[choice-1]

        if company not in data['Company'].values:
            ui.ai_response("Invalid company selected")
        else:
            break
    
    r_d_growth = data.loc[data['Company'] == company, 'R&D one-year growth (%)']
    
    plt.figure(figsize=(8,6))
    plt.title(f"R&D one-year Growth Percentage ({company})")
    plt.xlabel("Year Index")
    plt.ylabel("Growth Rate (%)")
    plt.plot(r_d_growth)
    plt.show()
elif 'Net Sales' in user_question:
    # Ask the user to select company for which Net sales one-year growth percentage is required
    companies = data['Company'].unique()

    while True:
        ui.ai_response("Select a Company:")
        for i, company in enumerate(companies):
            ui.ai_response(f"{i+1}. {company}")
        
pass
        company = companies[choice-1]

        if company not in data['Company'].values:
            ui.ai_response("Invalid company selected")
        else:
            break
    
    net_sales_growth = data.loc[data['Company'] == company, 'Net sales one-year growth (%)']
    
    plt.figure(figsize=(8,6))
    plt.title(f"Net Sales one-year Growth Percentage ({company})")
    plt.xlabel("Year Index")
    plt.ylabel("Growth Rate (%)")
    plt.plot(net_sales_growth)
    plt.show()
else:
    ui.ai_response("Invalid question. Please ask 'R&D vs Net Sales'+' '+ 'R&D growth of a company' or 'Net sales growth of a company'")
    
# Save final output to csv
data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
