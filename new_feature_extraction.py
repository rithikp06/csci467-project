import pandas as pd
import numpy as np

STOCKS = ["aapl", "amd", "amzn", "googl", "meta", "msft", "nflx", "nvda", "qcom", "tsla"]
DIR = "/Users/rithikpothuganti/cs467/csci467-project/data"

def assign_label(row):
    if row['target_return'] <= -0.1:
        # strong sell
        return 0
    elif -0.1 < row['target_return'] <= -0.05:
        # underweight
        return 1
    elif -0.05 < row['target_return'] <= 0.05:
        # neutral
        return 2
    elif 0.05 < row['target_return'] <= 0.1:
        # overweight
        return 3
    else:
        # strong buy
        return 4

def get_stock_df(file_path):
    df = pd.read_csv(file_path)
    df = df[::-1].reset_index()

    df['Close/Last'] = df['Close/Last'].apply(lambda x: float(str(x)[1:]))

    # Calculate returns over various time windows
    df['r_90d'] = df['Close/Last'].pct_change(62)
    df['r_180d'] = df['Close/Last'].pct_change(125)
    df['r_1y'] = df['Close/Last'].pct_change(250)
    df['r_2y'] = df['Close/Last'].pct_change(500)
    df['r_5y'] = df['Close/Last'].pct_change(1250)
    df['r_10y'] = df['Close/Last'].pct_change(2500)


    # Create a label column with the target return and buy/sell signal
    df['target_return'] = df['Close/Last'].pct_change(255).shift(-255)
    df['label'] = df.apply(lambda row: (1 if row['target_return'] > 0 else 0), axis=1)
    df['rating'] = df.apply(assign_label, axis=1)


    # Select the relevant columns for the training examples
    features = ['r_90d', 'r_180d', 'r_1y', 'r_2y', 'r_5y', 'r_10y']
    label = ['target_return', 'label', 'rating', 'Date', 'Close/Last']
    df_ml = df.loc[255:, features + label].dropna(subset=['target_return'])
    return df_ml

def get_all_stocks_df():
    dfs = []
    for stock in STOCKS:
        df_temp = get_stock_df(f"{DIR}/{stock}.csv")
        dfs.append(df_temp)
    return pd.concat(dfs, axis=0, ignore_index=True)

def featurize(x, d=1):
    if d == 1:
        return x
    else:
        features = x
        for i in range(2,d+1):
            features = np.concatenate((features,x**i), axis=1)
        return features

def read_data(X_df, Y_df):
    return X_df.to_numpy(), Y_df['target_return'].to_numpy()

def read_logistic_data(X_df, Y_df):
    return X_df.to_numpy(), Y_df['label'].to_numpy()

def read_buy_rating_data(X_df, Y_df):
    return X_df.to_numpy(), Y_df['rating'].to_numpy()

if __name__ == "__main__":
    df = get_all_stocks_df()
    print(df)
