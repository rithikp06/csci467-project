import pandas as pd
import numpy as np


def first_last(df):
    return df.iloc[[0,-1]]

def last(df):
    return df.iloc[[-1]]

class Feature_Extractor:
    def __init__(self, prices_file_path):
        self.price_data = pd.read_csv(prices_file_path)
        self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
        self.price_data['Open'] = self.price_data['Open'].apply(lambda x: float(str(x)[1:]))
        self.price_data['High'] = self.price_data['High'].apply(lambda x: float(str(x)[1:]))
        self.price_data['Low'] = self.price_data['Low'].apply(lambda x: float(str(x)[1:]))
        self.price_data = self.price_data[::-1].reset_index()

        self.monthly_data = self.get_monthly_data()
        self.monthly_return_data = self.get_monthly_return_data()

    def get_all_data(self):
        return self.price_data

    def get_price_data_by_range(self, start_date, end_date):
        return self.price_data.loc[(self.price_data['Date'] <= pd.to_datetime(end_date))  & (self.price_data['Date'] >= pd.to_datetime(start_date))]

    # percentage of days where price is greater than or less than target price
    def get_price_probability(self, greater, payoff_percentage, price_data, price_type):
        payoff_percentage /= 100
        target_price = (1 + payoff_percentage) * np.average(price_data[price_type])
        if greater:
            return np.sum(price_data[price_type] >= target_price) / len(price_data)
        else:
            return np.sum(price_data[price_type] <= target_price) / len(price_data)
       
    def get_monthly_data(self):
        monthly_data = self.price_data.copy()
        monthly_data['Date'] = monthly_data['Date'].apply(lambda x: x.strftime('%Y-%m'))
        monthly_data = monthly_data.groupby('Date').mean()
        return monthly_data

    # average price per month for last n months
    def get_n_months_data(self, end_date, n_months):
        end_date = pd.to_datetime(end_date)
        start_date = end_date - pd.DateOffset(months=n_months)
        return self.monthly_data.loc[(self.monthly_data.index <= end_date.strftime('%Y-%m')) & (self.monthly_data.index > start_date.strftime('%Y-%m'))]
    

    def get_monthly_return_data(self):
        monthly_return_data = self.price_data.copy()
        monthly_return_data['Date'] = monthly_return_data['Date'].apply(lambda x: x.strftime('%Y-%m'))
        monthly_return_data = monthly_return_data.groupby('Date').apply(first_last)
        monthly_return_data["Return"] = monthly_return_data["Open"].pct_change()
        monthly_return_data = monthly_return_data.reset_index(drop=True)
        monthly_return_data = monthly_return_data.groupby('Date').apply(last)
        return monthly_return_data
    
    def get_n_months_return_data(self, end_date, n_months):
        end_date = pd.to_datetime(end_date)
        start_date = end_date - pd.DateOffset(months=n_months)
        return self.monthly_return_data.loc[(self.get_monthly_return_data()['Date'] <= end_date.strftime('%Y-%m')) & (self.get_monthly_return_data()['Date'] > start_date.strftime('%Y-%m'))]

    """
    label is average price of the last month in price_data
    features is average price for each of the 4 months prior
    need to update this to be monthly return
    """
    def get_labels_and_features(self, price_data):
        features = []
        for i in range(1, len(price_data)):
            features.append(price_data.iloc[i]['Return'])
        label = price_data.iloc[-1]['Return']
        return label, features