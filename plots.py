import pandas as pd
import matplotlib.pyplot as plt


def RSI_n(prices: pd.Series, n: int = 14, ema: bool = False) -> pd.Series:
   """
   Relative Strength Index (RSI) calculated with a period of 'n'
   Using the formula RSI = 100 - 100 / (100 + RS)
   Where RS = GAIN_avg / LOSS_avg
   """
   prices = pd.to_numeric(prices, errors='coerce')
   delta = prices.diff()

   gain = delta.where(delta > 0, 0.0)
   loss = - delta.where(delta < 0, 0.0)

   if ema:
    avg_gain = gain.ewm(span=n, adjust=False).mean()
    avg_loss = loss.ewm(span=n, adjust=False).mean()
   else:
    avg_gain = gain.rolling(window=n, min_periods=n).mean()
    avg_loss= loss.rolling(window=n, min_periods=n).mean()

   rs = avg_gain / avg_loss.replace(0, pd.NA)

   rsi = 100 - 100 / (1 + rs)
   rsi = rsi.fillna(100)

   return rsi.clip(lower=0, upper=100)


### Preprocessing STOCK ###

def main(STOCK: str = 'Investor', filepath: str = "data/Investor.csv", all_plots: bool = True):

    """
    Main function of plot generation.
    Plots relevant plots with data from the stock file.

    Calculates Discount/Premium, RSI and Return rate
    """


    exclude_cols = ['Rabatt/Premie', 'Genomsnittsrabatt senaste 100 handelsdagarna', 'Nuvarande rabatt minus snitt', 'Avkastning 200 handelsdagar', 'OMX Date']
    numeric_cols = ['PRIS', 'SUBSTANSVÄRDE', 'BERÄKNAT_SUBSTANSVÄRDE', 'Index Value', 'Avkastning OMXS#=']
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

    cutoff_date = '2024-12-11'
    df['Investor Date'] = pd.to_datetime(df['Investor Date'], errors='coerce')
    df = df[df['Investor Date'] < cutoff_date]

    # Rabatt/premie: Ber.substansvärde / pris

    df['RABATT/PREMIE'] = (df['BERÄKNAT_SUBSTANSVÄRDE'] - df['PRIS']) / df['PRIS']
    df['AVKASTNING'] =(df['PRIS'].shift(-200) - df['PRIS']) / df['PRIS']
    df['AVKASTNING - OMXS'] = df['AVKASTNING'] - df['Avkastning OMXS#=']*0.01

    df['AVKASTNING'] = (
        df['AVKASTNING']
        .astype(str)
        .str.replace(',', '.', regex=False)   # convert commas to dots
        .str.replace(' ', '', regex=False)    # remove spaces
        .pipe(pd.to_numeric, errors='coerce') # convert to float safely
    )

    df['RSI'] = RSI_n(df['PRIS'], n= 14, ema=True)

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

    plt.scatter(df['RABATT/PREMIE'], df['RSI'])
    plt.xlabel('Discount/Premium')
    plt.ylabel('RSI')
    plt.title(f'Discount/Premium against RSI for {STOCK}')
    plt.show()

    ### BOXPLOTS ###

    if all_plots:
        df = df.dropna(subset=['AVKASTNING'])
        plt.boxplot([df['AVKASTNING'], df['Avkastning OMXS#=']*0.01], labels=[f'Return rate {STOCK}', f'Return rate OMXS'])
        plt.title(f'Return rate boxplot for {STOCK}')
        plt.ylabel('Return rate (%)')
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
    if all_plots:
        plt.plot(df['Investor Date'], df['RABATT/PREMIE'])
        plt.xlabel('Date')
        plt.ylabel('Discount/Premium')
        plt.show()

        plt.plot(df['Investor Date'], df['RSI'])
        plt.axhline(70, linestyle='--', linewidth=1, label='70')
        plt.axhline(30, linestyle='--', linewidth=1, label='30')
        plt.legend()
        plt.show()


        plt.plot(df['Investor Date'], df['AVKASTNING'], label= f'{STOCK} return rate')
        plt.plot(df['Investor Date'], df['Avkastning OMXS#=']*0.01, label= f'OMXSreturn rate')
        plt.legend()
        plt.show()


if __name__ ==  '__main__':

#    main('Investor', 'data/Investor.csv')
   
   main('Industrivärlden', 'data/Industrivarden_vanlig2.csv')