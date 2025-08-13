import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
 
# Load the dataset
df = pd.read_csv('train.csv')
# Display the first few rows of the dataset
print(df.head())
print(df.tail())
# Display the shape of the dataset
print(df.shape)
print(df.info())
# Display the summary statistics of the dataset 
print(df.describe())
#data cleaning
# Check for missing values  
df.isnull().sum()
# Drop rows with missing values 
df.dropna(inplace=True)
#check for duplicates
df.duplicated().sum()
# Drop duplicate rows
df.drop_duplicates(inplace=True)
#surviving passenger count
plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df, palette='Set1')
plt.title('Count of Survived vs Not Survived')  
plt.xticks([0, 1], ['Didn\'t Survive', 'Survived'])
plt.xlabel('Survival Status')
plt.ylabel('Count')
#survival on gender basis
survived_df = df[df['Survived'] == 1]
plt.subplot(2, 2, 2)
sns.countplot(x='Sex', data=survived_df, palette='Set2')
plt.title('male/female Survived Count')
plt.xlabel('survival Status')
plt.ylabel('Count')
#survival on age basis
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase vertical and horizontal spacing

plt.subplot(2, 2, 3)
sns.histplot(x='Age', data=survived_df, palette='viridis')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
#survival on class basis
plt.subplot(2, 2, 4)
ax = plt.gca()  # Get current axes
class_counts = survived_df['Pclass'].value_counts().sort_index()
colors = sns.color_palette('Set1', n_colors=3)
labels = ['Class 1', 'Class 2', 'Class 3']
patches, texts, autotexts = plt.pie(class_counts, labels=labels, startangle=140, colors=colors, autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.title('Survived Passengers by Class (Pie Chart)')
plt.legend(patches, labels, loc='best', title='Passenger Class', frameon=True)
ax.patch.set_edgecolor('black')  # Add black border to the subplot
ax.patch.set_linewidth(2)        # Set border thickness
plt.tight_layout()
plt.show()

#Age Distribution of Passengers
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
age_bins = list(range(0, 101, 10))
age_labels = [f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(len(age_bins)-1)]
survived_df['AgeGroup'] = pd.cut(survived_df['Age'], bins=age_bins, labels=age_labels, right=False)
age_group_counts = survived_df['AgeGroup'].value_counts().sort_index()
sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='viridis')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')


#gender distribution of passengers
plt.subplot(1, 2, 2)
gender_counts= df['Sex'].value_counts().sort_index()
labels = ['Male','female']
plt.pie(gender_counts, startangle=140, autopct='%1.1f%%', colors=sns.color_palette('bright6', n_colors=2))
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.title('Sex Distribution of Passengers')
plt.legend(labels, loc='best', title='Genders', frameon=True)
plt.tight_layout()
plt.show()

#fare distribution of passengers by class
plt.figure(figsize=(15, 6))

sns.barplot(x='Pclass', y='Fare', data=df, palette='viridis')
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')  
plt.show()

#converting sex column to numerical values
df['sexnum'] = df['Sex'].map({'male': 0, 'female': 1})
#correlation matrix between survived, sexnum, age
correlation_matrix = df[['Survived', 'sexnum', 'Age']].corr()
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Survival')
plt.xlabel('Features')
plt.ylabel('Features')

#converting pclass column to numerical values
df['pclassnum'] = df['Pclass'].astype(int)
#converting cabin column to numerical values
df['Cabin'] = df['Cabin'].fillna('Unknown')  # Fill missing values with 'Unknown'
df['Cabin'] = df['Cabin'].str.extract('([A-Za-z])')[0]  # Extract the first letter of the cabin
df['Cabin'] = df['Cabin'].map(lambda x: ord(x) - ord('A') + 1 if pd.notnull(x) else 0)  # Convert letters to numerical
# Correlation matrix between pclass, cabin, fare
correlation_matrix1 = df[['Cabin', 'pclassnum', 'Fare']].corr()
plt.subplot(1, 2, 2)
sns.heatmap(correlation_matrix1, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Passenger Class and Fare')
plt.xlabel('Features')
plt.ylabel('Features')  
plt.tight_layout()
plt.show()

