import pandas as pd
import numpy as np

def create_dataset(symbol: str, max_samples: int = None):
    print("Loading data into the memory...")
    binance_realized = pd.read_csv('data/binanceRealizedFundings.csv')
    bybit_realized = pd.read_csv('data/bybitRealizedFundings.csv')
    funding_rates = pd.read_csv('data/nextFundingRates.csv')
    print("Data loaded into the memory")
    print(f"Prepairing dataset for symbol: [{symbol}]")
    
    funding_rates = funding_rates.loc[(funding_rates['exchange'] == 'BINANCEF') & (funding_rates['symbol'] == symbol)].copy()
    funding_rates['timestamp'] = pd.to_datetime(funding_rates['timestamp'], unit='ms')

    bybit_realized.rename(columns=lambda cn: cn.split('_')[0], inplace=True)
    bybit_realized = bybit_realized[['timestamp', symbol]]
    bybit_realized['timestamp'] = pd.to_datetime(bybit_realized['timestamp'])
    bybit_realized = bybit_realized.loc[bybit_realized[symbol].notna()]
    bybit_realized = bybit_realized.reset_index()

    binance_realized.rename(columns=lambda cn: cn.split('_')[0], inplace=True)
    binance_realized = binance_realized[['timestamp', symbol]]
    binance_realized['timestamp'] = pd.to_datetime(binance_realized['timestamp'])
    binance_realized = binance_realized.loc[binance_realized[symbol].notna()]
    binance_realized = binance_realized.reset_index()

    last_funding_ts = binance_realized.iloc[0]['timestamp'] + pd.DateOffset(hours=-8)
    target_indices = np.linspace(-1, 1, 400)
    X, y = [], []

    dataset = pd.DataFrame()
    for index, binance_realized_row in binance_realized.iterrows():
        if index == 0: #skip the first row as no previous cycle
            continue
        if index == len(binance_realized) -1:
            break

        previous_realized = binance_realized.iloc[index-1]
        current_realized = binance_realized.iloc[index]
        next_realized = binance_realized.iloc[index+1]

        current_funding_cycle_end_ts = current_realized['timestamp']
        # print("Current cycle:", index, f"{str(last_funding_ts)} - {str(current_funding_cycle_end_ts)}")
        
        #Query the funding rate estimates for the current cycle
        current_funding_cycle_estimated_rates = funding_rates.query(f"timestamp > '{last_funding_ts}' & timestamp < '{current_funding_cycle_end_ts}'").copy()

        # print(f"Processing symbol: {symbol}")
        #Drop empty cycles
        if current_funding_cycle_estimated_rates.empty:
            # print("Empty")
            continue

        # #Drop samples with too few entries
        if len(current_funding_cycle_estimated_rates) < 200:
            # print(f"Too few entry in the funding cycle [{len(current_funding_cycle_estimated_rates)}]")
            continue

        bybit_realized_row = bybit_realized.loc[(bybit_realized['timestamp'] == previous_realized['timestamp'])]
        bybit_realized_value = bybit_realized_row[symbol].iloc[0]
        if bybit_realized_value is None:
            # print("No bybit realized funding for the current cycle")
            continue

        original_indicies = np.linspace(-1, 1, len(current_funding_cycle_estimated_rates))
        current_funding_cycle_estimates_interpolated = pd.DataFrame({"nextFundingRate": np.interp(target_indices, original_indicies, current_funding_cycle_estimated_rates['nextFundingRate'].to_list())})
        
        current_funding_cycle_estimates_interpolated['rolling_mean_12'] = current_funding_cycle_estimates_interpolated['nextFundingRate'].rolling(window=12).mean()
        current_funding_cycle_estimates_interpolated['rolling_std_12'] = current_funding_cycle_estimates_interpolated['nextFundingRate'].rolling(window=12).std()
        current_funding_cycle_estimates_interpolated['rolling_mean_24'] = current_funding_cycle_estimates_interpolated['nextFundingRate'].rolling(window=24).mean()
        current_funding_cycle_estimates_interpolated['rolling_std_24'] = current_funding_cycle_estimates_interpolated['nextFundingRate'].rolling(window=24).std()
        current_funding_cycle_estimates_interpolated['previous_realized'] = previous_realized[symbol]
        current_funding_cycle_estimates_interpolated['ticks_until_realization'] = range(399, -1, -1)
        current_funding_cycle_estimates_interpolated['bb_previous_realized'] = bybit_realized_value

        current_funding_cycle_estimates_interpolated.bfill(inplace=True)


        if current_funding_cycle_estimates_interpolated.isnull().values.any():
            print(current_funding_cycle_estimates_interpolated.isnull().values) 
            print(current_funding_cycle_estimates_interpolated.loc[current_funding_cycle_estimates_interpolated['rolling_std_12'].isna()])
            exit(1)


        target = binance_realized.iloc[index+1][symbol]

        dataset = pd.concat([dataset, current_funding_cycle_estimates_interpolated], ignore_index=True)

        y = y + [target]*400

        last_funding_ts = current_funding_cycle_end_ts

        if max_samples is not None and len(dataset) >= max_samples:
            break
    
    dataset.reset_index(inplace = True)
    dataset = dataset.drop(['index'], axis=1)

    print(f"Dataset ready for symbol: [{symbol}]")
    return dataset, y