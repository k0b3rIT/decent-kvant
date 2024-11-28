Decent kvant
=====

This solution is using a GradientBoostingRegressor to estimate the next funding rate.

The model was trained on these symbols:
- BTCUSDT
- ETHUSDT
- ADAUSDT
- DOTUSDT
- 1INCHUSDT
- AAVEUSDT
- BNBUSDT
- ENJUSDT
- DOGEUSDT

These includes 601.200 samples each with 8 features\
The training took around 1.5 hours

Features in each sample:
- nextFundingRate (Estimated funding rate)
- rolling_mean_12
- rolling_std_12
- rolling_mean_24
- rolling_std_24
- previous_realized (Realized rate in the last cycle on Binance)
- ticks_until_realization (how far is the current sample is from the realization)
- bb_previous_realized (Realized rate in the last cycle on Bybit)

How to prepair the code for execution
--------
1. Clone the repo

2. Create a python 3.10 virtualenv
```
pyenv virtualenv 3.10.11 decent_kvant
```

1. Activate the new env
```
pyenv activate decent_kvant
```

1. Install the required pip packages
```
pip install -r requirements.txt
```

1. Copy the csv-s into the `data` folder
```
data
├── binanceRealizedFundings.csv
├── bybitRealizedFundings.csv
└── nextFundingRates.csv
``` 

How to execute the prediction
---------

2 ways to do that

### Execute the model directly

The model weights got saved into the `gbr_funding_rate.pickle` file so you can just run the prediction.

This will run the prediction with a default symbol (`MKRUSDT`) to change the symbol you can modify the code.
```
python gbr.py --inference
```

### Execute with a jupyter notebook

#### Dedicated jupyter server
install an additional package
```
pip install jupyter
```

Start the jupyter notebook
```
jupyter notebook
```

Select the `gbr_notebook.ipynb` file, and run the steps

#### In VSCode (if you have the jupyter plugin installed)
Select the `gbr_notebook.ipynb` file, and run the steps

How to retrain the model
---------
Modify the code to adjust the desired symbols to learn on.
Then execute the following command. 
```
python gbr.py --train
```
