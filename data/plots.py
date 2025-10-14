import pandas as pd
import matplotlib.pyplot as plt


### Preprocessing STOCK ###

STOCK = 'Investor'

exclude_cols = ['Rabatt/Premie', 'Genomsnittsrabatt senaste 100 handelsdagarna', 'Nuvarande rabatt minus snitt', 'Avkastning 200 handelsdagar', 'OMX Date']

df = pd.read_csv("data/Investor.csv", usecols=lambda col: col not in exclude_cols)

numeric_cols = ['PRIS', 'SUBSTANSVÄRDE', 'BERÄKNAT_SUBSTANSVÄRDE', 'Index Value', 'Avkastning OMXS#=']




for col in numeric_cols:
   df[col] = (
        df[col]
        .astype(str)                 # make sure it's a string
        .str.replace(' ', '', regex=False)  # remove spaces inside numbers
        .str.replace(',', '.', regex=False)
        .str.replace('%', '', regex=False) # replace decimal commas with dots
        .pipe(pd.to_numeric, errors='coerce')  # safely convert to float
    )

df['Investor Date'] = pd.to_datetime(df['Investor Date'], errors='coerce')

# print(df.head())
# print(df.dtypes)


cutoff_date = '2024-12-11'

df = df[df['Investor Date'] < cutoff_date]


num_subset = df[numeric_cols]


# Rabatt/premie: Ber.substansvärde / pris

df['RABATT/PREMIE'] = (df['BERÄKNAT_SUBSTANSVÄRDE'] - df['PRIS']) / df['PRIS']
df['AVKASTNING'] =(df['PRIS'].shift(-200) - df['PRIS']) / df['PRIS']

df['AVKASTNING - OMXS'] = df['AVKASTNING'] - df['Avkastning OMXS#=']*0.01


### CORRELATION PLOTS ###

plt.scatter(df['RABATT/PREMIE'], df['AVKASTNING - OMXS'])
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Discount/Premium')
plt.ylabel('Return rate adjusted for OMXS')
plt.title(f'Discount/Premium against adjusted return rate for {STOCK}')
plt.show()


plt.scatter(df['RABATT/PREMIE'], df['AVKASTNING'])
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Discount/Premium')
plt.ylabel('Return rate')
plt.title(f'Discount/Premium against return rate for {STOCK}')
plt.show()

### BOXPLOTS ###

plt.boxplot(df['AVKASTNING'])
plt.title(f'Return rate boxplot for {STOCK}')
plt.show()

plt.show()
plt.boxplot(df['AVKASTNING - OMXS'])
plt.title(f'Adjusted return rate boxplot for {STOCK}')
plt.show()


plt.boxplot(df['RABATT/PREMIE'])
plt.title(f'Discount/premium boxplot for {STOCK}')
plt.show()


### HISTOGRAMS ###

plt.hist(df['RABATT/PREMIE'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Discount / Premium ')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f'Discount/Premium histogram for {STOCK}')
plt.show()

### Plots over time ###

plt.plot(df['Investor Date'], df['RABATT/PREMIE'])
plt.xlabel('Date')
plt.ylabel('Discount/Premium')
plt.show()
