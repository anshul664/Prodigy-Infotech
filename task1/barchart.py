import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('worldpopulationdata.csv')

df.head(10)
df.tail(10)
df.info()
df.describe()
#checking duplicate
df.duplicated().sum()
#checking missing values
df.isnull().sum()
#checking unique values
print(df['Country Name'].unique())
print("\n total number of unique countries:",df['Country Name'].nunique())
print(df['Country Code'].unique())
print("\n total number of unique countries:",df['Country Code'].nunique())
#dropping unnecessary columns
df.drop(columns=['Series Name', 'Country Name'], axis=1,inplace=True)

#Extraction of top 10 countries with wrt total population
Total_Population = df[df['Series Code'] == 'SP.POP.TOTL']
Total_Population_sorted = Total_Population.sort_values(by='2022', ascending=False).head(10)
#get the top 10 countries with highest population
Total_top_ten_countries = Total_Population_sorted.head(10)
print(Total_top_ten_countries)

#Top 10 countries with highest population in 2022 and 2016
#create bar plot
plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
sns.barplot(x='Country Code', y='2022', data=Total_top_ten_countries, palette='viridis')
plt.title('Top 10 Countries by Population in 2022', fontsize=10)
plt.xlabel('country', fontsize=10)
plt.ylabel('Population', fontsize=10)



plt.subplot(1,2,2)
sns.barplot(x='Country Code', y='2016', data=Total_top_ten_countries, palette='coolwarm')
plt.title('Top 10 Countries by Population in 2016', fontsize=10)

plt.xlabel('country', fontsize=10)
plt.ylabel('Population', fontsize=10)
plt.show()

# Reshape the dataframe for grouped barplot
df_melted = Total_top_ten_countries.melt(id_vars='Country Code', value_vars=['2016', '2022'],var_name='Year', value_name='Population')
# Create grouped bar plot
plt.figure(figsize=(15, 6))
custom_palette = [
    sns.color_palette("coolwarm", 8)[2],  # Soft cool
    sns.color_palette("viridis", 8)[6]    # Deep viridis
]
sns.barplot(x='Country Code', y='Population', hue='Year', data=df_melted, palette=custom_palette)

plt.title('Top 10 Countries by Population in 2016 and 2022', fontsize=14)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.legend(title='Year')
plt.tight_layout()
plt.show()
# Extraction of bottom 10 countries with wrt total population
Total_Population_sorted1= Total_Population.sort_values(by='2022', ascending=True).head(10)
#get the bottom 10 countries with lowest population 
Total_bottom_ten_countries = Total_Population_sorted1.head(10)
print(Total_bottom_ten_countries)

#Bottom 10 countries with lowest population in 2022 and 2016
#create bar plot    
plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
sns.barplot(x='Country Code', y='2022', data=Total_bottom_ten_countries, palette='viridis')
plt.title('Bottom 10 Countries by Population in 2022', fontsize=10) 
plt.xlabel('country', fontsize=10)
plt.ylabel('Population', fontsize=10)   
plt.subplot(1,2,2)
sns.barplot(x='Country Code', y='2016', data=Total_bottom_ten_countries, palette='coolwarm')
plt.title('Bottom 10 Countries by Population in 2016', fontsize=10)
plt.xlabel('country', fontsize=10)
plt.ylabel('Population', fontsize=10)
plt.show()

# Reshape the bottom 10 data using melt
df_bottom_melted = Total_bottom_ten_countries.melt(
    id_vars='Country Code', 
    value_vars=['2016', '2022'], 
    var_name='Year', 
    value_name='Population'
)
# Create the merged grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Country Code', 
    y='Population', 
    hue='Year', 
    data=df_bottom_melted, 
    palette='Set2'  # You can change to 'Paired', 'colorblind', or custom palette
)
plt.title('Bottom 10 Countries by Population in 2016 and 2022', fontsize=14)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.legend(title='Year')
plt.tight_layout()
plt.show()


#extraction of top 10 countries with wrt highest male population
male_population = df[df['Series Code'] == 'SP.POP.TOTL']
#sort the data based on male population in 2022
male_population_sorted=male_population.sort_values(by='2022', ascending=False).head(10)
#get the top 10 countries with highest  
male_top_ten_countries= male_population_sorted.head(10)
print(male_top_ten_countries)

#extraction of top 10 countries with wrt highest female population
female_population = df[df['Series Code'] == 'SP.POP.TOTL']
#sort the data based on female population in 2022
female_population_sorted = female_population.sort_values(by='2022', ascending=False).head(10)
#get the top 10 countries with highest  
female_top_ten_countries= female_population_sorted.head(10)
print(female_top_ten_countries)

#Creating bar plot for top 10 countries with highest male and female population in 2022 and 2016
#male population
plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
sns.barplot(x='Country Code', y='2022', data= male_top_ten_countries, palette='viridis')
plt.title('Top 10 Countries with highest male population in 2022', fontsize=10)
plt.xlabel('country', fontsize=10)  
plt.ylabel('male population', fontsize=10)
plt.subplot(1,2,2)
sns.barplot(x='Country Code', y='2016' , data=male_top_ten_countries, palette='coolwarm')
plt.title('Top 10 Countries with highest male population in 2016', fontsize=10)
plt.xlabel('country', fontsize=10)  
plt.ylabel('male population', fontsize=10)
plt.show()

# Reshape the male population data
df_male_melted = male_top_ten_countries.melt(
    id_vars='Country Code',
    value_vars=['2016', '2022'],
    var_name='Year',
    value_name='Male Population'
)

# Create grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Country Code',
    y='Male Population',
    hue='Year',
    data=df_male_melted,
    palette='colorblind'  
)

plt.title('Top 10 Countries by Male Population in 2016 and 2022', fontsize=14)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Male Population', fontsize=12)
plt.legend(title='Year')
plt.tight_layout()
plt.show()

#female population
plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
sns.barplot(x='2022' , y='Country Code', data=female_top_ten_countries, palette='viridis')
plt.title('Top 10 Countries with highest female population in 2022', fontsize=10)
plt.xlabel('country', fontsize=10)
plt.ylabel('female population', fontsize=10)
plt.subplot(1,2,2)
sns.barplot(x='2016', y='Country Code', data=female_population_sorted, palette='coolwarm')
plt.title('Top 10 Countries with highest female population in 2016', fontsize=10)
plt.xlabel('country', fontsize=10)
plt.ylabel('female population', fontsize=10)
plt.show()
# Merge/reshape the data
df_female_melted = female_top_ten_countries.melt(
    id_vars='Country Code',
    value_vars=['2016', '2022'],
    var_name='Year',
    value_name='Female Population'
)

# Create horizontal grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    y='Country Code',
    x='Female Population',
    hue='Year',
    data=df_female_melted,
    palette='Paired'
)

plt.title('Top 10 Countries by Female Population in 2016 and 2022', fontsize=14)
plt.xlabel('Female Population', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.legend(title='Year')
plt.tight_layout()
plt.show()


