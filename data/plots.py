import pandas as pd



df = pd.read_csv("data/Investor.csv")


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.')

print(df.head())
print(df.dtypes)
