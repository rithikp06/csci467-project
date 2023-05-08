import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

STOCKS = ["aapl", "amd", "amzn", "googl", "meta", "msft", "nflx", "nvda", "qcom", "tsla"]
DIR = "data"

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


def normalize(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1,1))
    return data_scaled, scaler

def simple_rnn(X_train, y_train, X_test):
    model = Sequential()
    model.add(SimpleRNN(32, return_sequences = True))
    model.add(SimpleRNN(32))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=150, verbose=0)
    predictions = model.predict(X_test)
    return model, predictions

def main():
    df = get_all_stocks_df()

    features = ['r_90d', 'r_180d', 'r_1y', 'r_2y', 'r_5y', 'r_10y']
    label = ['target_return', 'label', 'Date', 'Close/Last', 'rating']

    X = df[features]
    X.fillna(0, inplace=True)
    y = df[label]
    # Define the split indices
    train_split_idx = int(len(X) * 0.7)
    dev_split_idx = int(len(X) * 0.85)

    # Train split
    X_train_df, y_train_df = X.iloc[:train_split_idx], y.iloc[:train_split_idx]
    train_dates = df["Date"].iloc[:train_split_idx]

    # Dev (validation) split
    X_dev_df, y_dev_df = X.iloc[train_split_idx:dev_split_idx], y.iloc[train_split_idx:dev_split_idx]
    dev_dates = df["Date"].iloc[train_split_idx:dev_split_idx]

    # Test split
    X_test_df, y_test_df = X.iloc[dev_split_idx:], y.iloc[dev_split_idx:]
    test_dates = df["Date"].iloc[dev_split_idx:]

    DEGREE = 1
    X_train, y_train = read_data(X_train_df, y_train_df)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    X_train = featurize(X_train, d=DEGREE)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    X_dev, y_dev = read_data(X_dev_df, y_dev_df)
    y_dev = np.reshape(y_dev, (y_dev.shape[0], 1))
    X_dev = featurize(X_dev, d=DEGREE)
    X_dev = np.reshape(X_dev, (X_dev.shape[0], X_dev.shape[1], 1))

    X_test, y_test = read_data(X_test_df, y_test_df)
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    model, prediction = simple_rnn(X_train, y_train, X_dev)
    print(y_dev)



if __name__ == '__main__':
    main()
