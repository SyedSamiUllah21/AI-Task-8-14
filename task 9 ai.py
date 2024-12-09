import pandas as pd

data = pd.read_csv(r'C:\Users\syeds\Downloads\tasks.csv')

print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.describe())
print(data.shape)
print(data.columns)
