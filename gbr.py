import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle
import argparse
from matplotlib import pyplot as plt

from gbr_data import create_dataset


def training():

    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', '1INCHUSDT', 'AAVEUSDT', 'BNBUSDT', 'ENJUSDT', 'DOGEUSDT']

    X, y = pd.DataFrame(), []

    for symbol in symbols:
        Xc, yc = create_dataset(symbol)
        X = pd.concat([X, Xc], ignore_index=True)
        y = y + yc

    print(X.shape)
    print(len(y))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(random_state=42)

    # Hyperparameter tuning (Optional)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    grid_search = GridSearchCV(gbr, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    pickle.dump(best_model, open('gbr_funding_rate.pickle', "wb"))

    # Evaluate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")


def inference(symbol:str ='MKRUSDT'):

    start_sample = 0

    X, y = create_dataset(symbol)
    print(X.shape)
    print(len(y))
    
    rmse_chart_data = []
    for si in range(start_sample*400, len(X), 400):

        test_set = X.iloc[si:si+400]

        loaded_model = pickle.load(open('gbr_funding_rate.pickle', "rb"))
        pred = loaded_model.predict(test_set)
        # plt.plot(test_set['nextFundingRate'].tolist())
        # plt.plot(pred)
        # plt.plot(y[si:si+400])
        # plt.legend(['samples', 'pred', 'realized'])
        # plt.show()

        rmse_original = np.sqrt(mean_squared_error(test_set['nextFundingRate'].tolist(), y[si:si+400]))
        rmse_gbr = np.sqrt(mean_squared_error(pred, y[si:si+400]))
        rmse_chart_data.append([rmse_original, rmse_gbr])

    rmse_tp = np.array(rmse_chart_data).transpose().tolist()
    plt.plot(rmse_tp[0])
    plt.plot(rmse_tp[1])
    plt.legend(['original', 'gbr_model'])
    plt.title(f"RMSE comparison [{symbol}]")
    plt.show()

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()
    if args.train:
        print("Training the model")
        training()
    elif args.inference:
        print("Inference the model")
        inference()
    else:
        print("No action specified")

if __name__ == "__main__":
    __main__()