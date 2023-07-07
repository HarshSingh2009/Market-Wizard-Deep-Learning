from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


class DataPreprocessing():
    def __init__(self, data):
        self.close_column = data.reset_index()['close']

    def min_max_scaling(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.close_df = scaler.fit_transform(np.array(self.close_column).reshape(-1, 1))
        # Saving the Scaler for furthur use
        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
        return self.close_df

    def train_test_splitting(self):
        training_size = int(len(self.close_df)*0.8)
        test_size = int(len(self.close_df)-training_size)
        self.train_data, self.test_data = self.close_df[0:training_size, :], self.close_df[training_size:len(self.close_df)]
        return self.train_data, self.test_data
    
    def create_dataset(self, dataset, timestep=150):
        dataX, dataY = [], []
        for i in range(len(dataset)-timestep-1):
            a = dataset[i: i+timestep, 0]
            dataX.append(a)
            dataY.append(dataset[i+timestep, 0])
        return np.array(dataX), np.array(dataY)

    def reshape_X_train_test(self, X_train, X_test):
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        return X_train, X_test
