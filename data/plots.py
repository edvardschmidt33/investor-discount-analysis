import pandas as pd


### Preprocessing Investor ###
exclude_cols = ['Rabatt/Premie', 'Genomsnittsrabatt senaste 100 handelsdagarna', 'Nuvarande rabatt minus snitt', 'Avkastning 200 handelsdagar']

df = pd.read_csv("data/Investor.csv", usecols=lambda col: col not in exclude_cols)

numeric_cols = ['PRIS', 'SUBSTANSVÄRDE', 'BERÄKNAT_SUBSTANSVÄRDE', 'Index Value', 'Avkastning OMXS#=']

for col in numeric_cols:
   df[col] = (
        df[col]
        .astype(str)                 # make sure it's a string
        .str.replace(' ', '', regex=False)  # remove spaces inside numbers
        .str.replace(',', '.', regex=False) # replace decimal commas with dots
        .pipe(pd.to_numeric, errors='coerce')  # safely convert to float
    )




print(df.head())
print(df.dtypes)


