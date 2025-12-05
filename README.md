# Mining the market: Premium and discount patterns of Swedish investment firms
This project investigates premium and discount patterns in Swedish
investment firms relative to their net asset value (NAV) and bench-
mark performance to the OMXS30 index. Using historical data
from IBindex and Nasdaq, to perform data preprocessing, pattern
mining (FP-growth), and train classification models (buy/wait) to
evaluate whether large discounts or premiums can help investors
systematically outperform the broader Swedish equity market.

For pre-pressesing results as well as tables with results from our two approaches see `figs`- folder. For model results of the first approach see model.ipynb, and results for the second approach pd_model.ipynb. For more pre-processing plots se `plots.py` and `presentation_plots.py`

### Replicating the project

To reproduce the results of the project, or retry it with new/different data, a file containing data from IB-Index as well as Nasdaq should be uploaded in a `.csv` fromat in the `\data` folder. The point odf entry in the resposatory is to run the `preprocess.py` script (or alternativly the `plots.py`-script) before doing running any other scripts since they all demand the processed format of the file.
Run the notebooks `model.ipynb` and `pd_model.ipynb` for the first and second approach of the project.
