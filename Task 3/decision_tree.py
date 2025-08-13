import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
warnings.filterwarnings("ignore")

#read data
df = pd.read_csv('bank-additional.csv', delimiter=';')
df.rename(columns={'y':'deposit'}, inplace=True)
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.shape)
print(df.dtypes)
#check for duplicates
print(df.duplicated().sum())
# Check for missing values
print(df.isnull().sum())

cat_cols = df.select_dtypes(include=['object']).columns
print("Categorical columns:", cat_cols)
num_cols = df.select_dtypes(exclude=['object']).columns
print("Numerical columns:", num_cols)

#describe including object
print(df.describe(include='object'))
#visual using histogram
df.hist(figsize=(10,10), color='#00FFFF')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Split cat_cols into two halves
half = len(cat_cols) // 2
cat_cols_part1 = cat_cols[:half]
cat_cols_part2 = cat_cols[half:]

# Split into 3 roughly equal parts
n = len(cat_cols)
part_size = math.ceil(n / 3)

cat_cols_part1 = cat_cols[:part_size]
cat_cols_part2 = cat_cols[part_size:2*part_size]
cat_cols_part3 = cat_cols[2*part_size:]

def plot_cat_cols(columns, part_title=""):
    num_cols = 2
    num_rows = (len(columns) + 1) // num_cols
    plt.figure(figsize=(22, 6 * num_rows))

    for i, col in enumerate(columns):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(data=df, x=col, hue='deposit', palette='Set2')
        plt.title(f'Count of {col} by Deposit {part_title}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend(title='Deposit', loc='upper right', fontsize=10)

    plt.tight_layout(pad=4)
    plt.show()

# Plot 3 parts
plot_cat_cols(cat_cols_part1, "(Part 1)")
plot_cat_cols(cat_cols_part2, "(Part 2)")
plot_cat_cols(cat_cols_part3, "(Part 3)")

#make inter-quantile range
df.plot(kind='box', subplots=True, layout=(2,5), figsize=(20,10), color='#7b3f00')
plt.show()

column=df[['age','campaign','duration']]
q1=np.percentile(column, 25)
q2=np.percentile(column, 75)
iqr= q2-q1
lower_bound=q1-1.5*iqr
upper_bound=q2+1.5*iqr
df[['age','campaign','duration']]=column[(column>lower_bound)&(column<upper_bound)]
df.plot(kind='box', subplots=True, layout=(2,5), figsize=(20,10), color="#4f3110")
plt.show()

#exclude non-numeric column
num_df=df.drop(columns=cat_cols)
#compute correlational matrix
corr=num_df.corr()
print(corr)
#filter correlation
corr = corr[abs(corr)>=0.90]
sns.heatmap(corr,annot=True,cmap='Set3',linewidths=-0.2)
plt.show()

high_corr_col=['emp.var.rate','euribor3m','nr.employed']
#copy df data
df1=df.copy()
print(df1.columns)
print(df1.shape)

#use labelencoder
ab=LabelEncoder()
df_encoded=df1.apply(ab.fit_transform)
print(df_encoded)

df_encoded['deposit'].value_counts()

#independent variable 
x=df_encoded.drop('deposit',axis=1)
y=df_encoded['deposit']

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, random_state=1)


def eval_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy Score:', acc)
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n', cm)
    print('Classification Report\n', classification_report(y_test, y_pred))

def mscore(model):
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print('Training Score:', train_score)
    print('Testing Score:', test_score)
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10)
dt.fit(x_train, y_train)
mscore(dt)

ypred_dt = dt.predict(x_test)
print(ypred_dt)

eval_model(y_test, ypred_dt)

cn=['no','yes']
FileNotFoundError
dn=x_train.columns
print(dn)

plt.figure(figsize=(30, 20))  # Wider and taller figure
plot_tree(dt, 
          class_names=cn, 
          filled=True, 
          rounded=True,            # Makes boxes rounded
          feature_names=df.columns.tolist(),  # Optional: add feature names
          fontsize=10)             # Smaller font for clarity
plt.tight_layout()
plt.show()
dt1 = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=15)
dt1.fit(x_train, y_train)

mscore(dt1)

ypred_dt1 = dt1.predict(x_test)
eval_model(y_test, ypred_dt1)
plt.figure(figsize=(30, 20))  # Wider and taller figure
plot_tree(dt1, 
          class_names=cn, 
          filled=True, 
          rounded=True,            # Makes boxes rounded
          feature_names=df.columns.tolist(),  # Optional: add feature names
          fontsize=10)             # Smaller font for clarity
plt.tight_layout()
plt.show()
