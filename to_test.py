import os
import pandas as pd

files = ['test_data/full_indu_c.csv', 'test_data/full_inve_b.csv', 'test_data/full_lato_b.csv']

frame = pd.read_excel('test_data/full_OMXS30.xlsx')
print(frame.head())

frame['Date'] = pd.to_datetime(frame['Trade Date'])

# Sort oldest → newest to compute returns correctly
frame = frame.sort_values('Trade Date')

frame = frame.dropna()
# Calculate daily returns
frame['Avkastning OMXS#='] = frame['Index Value'].pct_change()

print(frame[['Date', 'Index Value', 'Avkastning OMXS#=']].head)
frame.to_csv('test_data/OMXS30.csv')

for f in files: 
    company = pd.read_csv(f)
    company['DATUM'] = pd.to_datetime(company['DATUM'])

    
    company = company[company['DATUM'].isin(frame['Date'])]
    company['Avkastning OMXS#='] = frame.loc[frame['Date'].isin(company['DATUM']), 'Avkastning OMXS#='].values
    company['Investor Date'] = company['DATUM'].values
    company['Index Value'] = frame.loc[frame['Date'].isin(company['DATUM']), 'Index Value'].values

    path = os.path.dirname(f)
    file_name = os.path.basename(f)
    base, ext = os.path.splitext(file_name)


    path_name = os.path.join(path, f'{base}_test{ext}')

    company.to_csv(path_name)