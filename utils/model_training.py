import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
import pickle
from sklearn.metrics import mean_squared_error
import math

class Model_Training():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def create_and_train_model(self):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(150, 1))))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dense(1))

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam')
    
    def train_model(self):
        history = self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=64, validation_data=(self.X_test, self.y_test), verbose=1)
        return self.model, history
    
    def predict_test_data(self):
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)
        # Get the scaler
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        train_rmse = math.sqrt(mean_squared_error(self.y_train, train_predict))
        test_rmse = math.sqrt(mean_squared_error(self.y_test, test_predict))
        return train_predict, test_predict, train_rmse, test_rmse
    
