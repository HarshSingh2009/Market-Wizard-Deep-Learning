from tiingo import TiingoClient
from datetime import datetime
import streamlit as st
import pandas as pd
import os 

def list_files():
    files = []
    for filename in os.listdir('D:\MachineLearning\Deep Learning\Stock Prediction and Forecasting -Stacked LSTM\Datasets'):
        if os.path.isfile(os.path.join('D:\MachineLearning\Deep Learning\Stock Prediction and Forecasting -Stacked LSTM', filename)):
            files.append(filename)
    return files

API_KEY = '045ee573e2f164700d7bb102340fdf3b23b8489c'

def get_today_date():
    today = datetime.now()
    year = today.year
    month = today.month
    day = today.day
    return datetime(year, month, day)

class DataIngestion():
    def __init__(self, stock_code):
        self.stock_code = stock_code
    
    def get_data_set(self):
        if f'{self.stock_code}.csv' not in list_files():
            st.text('Using Generated Data from Tiingo')
            config = {
                'api_key': API_KEY,
                'session': True
            }
            client = TiingoClient(config)
            end_date =  get_today_date() # Specify the desired end date
            start_date = end_date.replace(year=end_date.year - 10)  # Assuming 5 years of data

            df = client.get_dataframe(self.stock_code, frequency='daily', startDate=start_date, endDate=end_date)
            df.to_csv(f'./Datasets/{self.stock_code}.csv')
            df = pd.read_csv(f'./Datasets/{self.stock_code}.csv', index_col='date', parse_dates=True)            
            return df
        
        else:
            st.text('Using Pre-used')
            return pd.read_csv(f'./Datasets/{self.stock_code}.csv')