import pandas as pd
import matplotlib.pyplot as plt

# Read the input file into a DataFrame
data = pd.read_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/t.csv')

# Group the data by 'Country' and count the number of bands for each country and genre combination
genre_counts = data.groupby(['Country', 'Genre']).size().reset_index(name='Count')

# Pivot the data to get a graph where x-axis is the country, y-axis is the genre count
pivot_data = pd.pivot_table(genre_counts, index='Country', columns='Genre', values='Count', aggfunc='sum')
pivot_data.plot(kind='bar', figsize=(15,10))

# Add title and labels to the plot
plt.title('Graph of Genre by Country')
plt.xlabel('Country')
plt.ylabel('Genre Count')
cmap=plt.get_cmap('hot')

# Show the plot
plt.savefig('foo.png')

# Save the final output as '/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv'
data.to_csv('/home/kman/VS_Code/projects/AirplaneModeAI/work/doData_Output.csv', index=False)
