import numpy as np
import matplotlib.pyplot as plt
import pickle


class ModelFuturePredictor():
    def __init__(self, model, train, test, close_df, num_days) -> None:
        self.model = model
        self.train_data = train
        self.test_data = test
        self.close_df = close_df
        self.num_days = num_days
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
    
    def predict_FuturePrice(self):
        look_back = 150
        x_input = self.test_data[self.test_data.shape[0] - look_back:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []
        n_steps = 150  # Updated value for timesteps
        i = 0
        while i < self.num_days:
            if len(temp_input) > n_steps:
                x_input = np.array(temp_input[-n_steps:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i += 1
            else:
                x_input = np.array(temp_input)
                x_input = x_input.reshape((1, len(temp_input), 1))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.extend(yhat.tolist())
                i += 1
            
        return self.scaler.inverse_transform(lst_output)
    
    def show_table_future_predictions(self, lst_output, stock_code):
        table_list = []
        for row in lst_output:
            table_list.append(row)
        return table_list
