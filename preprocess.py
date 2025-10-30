import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np

def preprocess(file:str = 'Investor.csv'):


    """
    Implements preprocessing steps such as normalization, data integration and attribute name and language standardization
    Feature Engineering: adding features "DISCOUNT/PREMIUM","DISCOUNT/PREMIUM_NORM" ,"RETURN", "RETURN - OMXS", "DAILY_RETURN", "VOL_30D", "SHARPE" 

    
    Args:
        file (str): Name of the file to be preprocessed

    Returns:
        new_file (csv): {file}_preprossesed.csv saved in "data" folder.
    """

    filepath = os.path.join('data', file)
    exclude_cols = ['Rabatt/Premie', 'Genomsnittsrabatt senaste 100 handelsdagarna', 'Nuvarande rabatt minus snitt', 'Avkastning 200 handelsdagar', 'OMX Date', 'SUBSTANSVÄRDE']
    numeric_cols = ['PRIS', 'BERÄKNAT_SUBSTANSVÄRDE', 'Index Value', 'Avkastning OMXS#=']
    df = pd.read_csv(filepath, usecols=lambda col: col not in exclude_cols)


    for col in numeric_cols:
        df[col] = (
                df[col]
                .astype(str)                 # make sure it's a string
                .str.replace(' ', '', regex=False)  # remove spaces inside numbers
                .str.replace(',', '.', regex=False)
                .str.replace('%', '', regex=False) # replace decimal commas with dots
                .pipe(pd.to_numeric, errors='coerce')  # safely convert to float
            )
   

    # Rabatt/premie: Ber.substansvärde / pris


    df['DISCOUNT/PREMIUM'] = (df['BERÄKNAT_SUBSTANSVÄRDE'] - df['PRIS']) / df['PRIS']
    df['RETURN'] =(df['PRIS'].shift(-200) - df['PRIS']) / df['PRIS']
    df['RETURN - OMXS'] = df['RETURN'] - df['Avkastning OMXS#=']*0.01
    
    
    df['DISCOUNT/PREMIUM_ADJ'] = (df['PRIS'] - df['BERÄKNAT_SUBSTANSVÄRDE']) / df['BERÄKNAT_SUBSTANSVÄRDE']

    df['RETURN'] = (
        df['RETURN']
        .astype(str)
        .str.replace(',', '.', regex=False)   # convert commas to dots
        .str.replace(' ', '', regex=False)    # remove spaces
        .pipe(pd.to_numeric, errors='coerce') # convert to float safely
    )

    cutoff_date = '2024-12-11'
    df['Investor Date'] = pd.to_datetime(df['Investor Date'], errors='coerce')
    df = df[df['Investor Date'] < cutoff_date]

    df['DISCOUNT/PREMIUM_NORM'] =   (df['DISCOUNT/PREMIUM'] - df['DISCOUNT/PREMIUM'].min()) / (df['DISCOUNT/PREMIUM'].max() - df['DISCOUNT/PREMIUM'].min())

    ### Standadize naming system, switch to English
    
    df = df.rename(columns={
    'BERÄKNAT_SUBSTANSVÄRDE': 'CALCULATED_NAV',
    'PRIS': 'PRICE',
    'Index Value': 'INDEX_VALUE',
    'Avkastning OMXS#=': 'RETURN_OMXS',
    'Investor Date': 'DATE'
})
    


    # Adding Volatility
    WINDOW= 30
    annual_rf = 0.0
    trading_days = 252
    df['DAILY_RETURN'] = df['PRICE'].pct_change()
    rf_daily = (1 + annual_rf)**(1/trading_days) - 1
    # add rolling averages
    mean_r = df['DAILY_RETURN'].rolling(WINDOW, min_periods=5).mean()
    std_r  = df['DAILY_RETURN'].rolling(WINDOW, min_periods=5).std()

    df['VOL_30D'] = std_r * np.sqrt(trading_days)

    # Adding Sharpe quotient
    df['SHARPE_30D'] = ((mean_r - rf_daily) / std_r) * np.sqrt(trading_days)

    exp_mean = df['DAILY_RETURN'].expanding(min_periods=20).mean()
    exp_std  = df['DAILY_RETURN'].expanding(min_periods=20).std()

    df['VOL_EXP']    = exp_std * np.sqrt(trading_days)
    df['SHARPE_EXP'] = ((exp_mean - rf_daily) / exp_std) * np.sqrt(trading_days)
    df.loc[~np.isfinite(df['SHARPE_EXP']), 'SHARPE_EXP'] = np.nan



    directory = os.path.dirname(filepath)  # 'data'
    base, ext = os.path.splitext(file)

    # Create the new filename
    save_path = os.path.join(directory, f"{base}_preprocess{ext}")

    return df.to_csv(save_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    preprocess('Investor.csv')
    preprocess('Industrivarden_vanlig2.csv')
    preprocess('Latour.csv')