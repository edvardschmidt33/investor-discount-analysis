import pandas as pd
import matplotlib.pyplot as plt
import os


def presentation(file: str = 'Investor_preprocess.csv', STOCK: str = 'Investor', adjusted: bool = False):
    filepath = os.path.join('data', file)

    df = pd.read_csv(filepath)
    # Extract the year
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['YEAR'] = df['DATE'].dt.year
    discount_type = 'DISCOUNT/PREMIUM_ADJ' if adjusted else 'DISCOUNT/PREMIUM'


    # Create the scatter plot, color-coded by year
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        df[discount_type], 
        df['RETURN - OMXS'], 
        c=df['YEAR'],              # color by year
        cmap='viridis',            # color map ('plasma', 'coolwarm', etc.)
        alpha=0.8
    )

    # Add a colorbar showing which color corresponds to which year
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year')

    # Add labels and lines
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Discount/Premium')
    plt.ylabel('Return rate adjusted for OMXS')
    plt.title(f'Discount/Premium vs Adjusted Return Rate for {STOCK}')
    plt.savefig(f'figs/discount_vs_return_{STOCK}.png', dpi=300, bbox_inches='tight')
    plt.show()
    

    corr = df['DISCOUNT/PREMIUM'].corr(df['RETURN - OMXS'])
    print(f'--Correlation for {STOCK}--')
    print(f"Correlation between Discount/Premium and Return - OMXS for {STOCK}:", corr)
    print(df[['DISCOUNT/PREMIUM', 'RETURN - OMXS', 'RETURN']].corr())


if __name__=='__main__':
    presentation('Investor_preprocess.csv', 'Investor', adjusted= False)
    presentation('Industrivarden_vanlig2_preprocess.csv', 'Industrivärlden', adjusted= False)
    presentation('Latour_preprocess.csv', 'Latour', adjusted= False)