from utils.data_ingestion import DataIngestion
from utils.data_preprocessing import DataPreprocessing
from utils.model_training import Model_Training
from utils.future_prediction import ModelFuturePredictor

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Bidirectional
from keras.models import load_model

import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import tempfile

st.set_page_config(layout="wide")


def summary_to_text(model):
    layer_info = []
    for layer in model.layers:
        if isinstance(layer, Bidirectional):
            layer_type = 'Bidirectional ' + layer.forward_layer.__class__.__name__
            units = layer.forward_layer.units
            return_sequences = layer.forward_layer.return_sequences
            info = f'{layer_type} layer with {units} units and return_sequences={return_sequences}'
        elif isinstance(layer, LSTM):
            layer_type = layer.__class__.__name__
            units = layer.units
            return_sequences = layer.return_sequences
            info = f'{layer_type} layer with {units} units and return_sequences={return_sequences}'
        elif isinstance(layer, Dense):
            layer_type = layer.__class__.__name__
            units = layer.units
            info = f'{layer_type} layer with {units} unit'
        else:
            info = f'{layer.__class__.__name__} layer'
        layer_info.append(info)

    # Format layer information as plain text
    layer_text = '\n'.join(f'- {info}' for info in layer_info)

    # Display the layer information as plain text in Streamlit
    st.text(layer_text)


def load_stock_codes(filename):
    stock_codes = []
    with open(filename, 'r') as file:
        for line in file:
            stock_codes.append(line.strip())
    return stock_codes

# Load stock codes from the file
stock_codes = load_stock_codes('stock_codes.txt')

def delete_files_in_folder():
    # Get the list of files in the folder
    files = os.listdir('./Numpy Arrays/')

    # Iterate over each file and delete it
    for file_name in files:
        file_path = os.path.join('./Numpy Arrays/', file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    

def get_free_space(num) -> None:
    for i in range(0, num):
        st.title('')

# app

with st.sidebar:
    selected = option_menu(
        menu_title='Market-Wizard',
        menu_icon='🕵️',
        options=['Home', 'Load & preprocess Data', 'Model Training', 'Predict']
    )
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')


if selected == 'Home':    
    delete_files_in_folder()
           
# Home/Instruction Manual
    st.title("📈 Market-Wizard App")
    st.caption('Stock Market Prediction and Forecasting !!')
    st.markdown("Welcome to the Market-Wizard which is an Stock Prediction and Forecasting App ! This app helps you predict the future stock prices of various companies. It consists of several segments that guide you through the process of loading data, preprocessing, model training, and making predictions.")

    get_free_space(1)

    st.header("📚 Instructions Manual")
    st.markdown("Please follow the steps below to effectively use the app:")

    get_free_space(1)

    st.subheader("🔹 Step 1: Load and Preprocess Data")
    st.markdown("1. In the navigation bar, click on 'Load and Preprocess Data'.")
    st.markdown("2. Enter the stock code of the company you want to analyze.")
    st.markdown("3. The app will load the stock data from Tiingo and display it as a dataframe. 📊")
    st.markdown("4. It will then show charts representing the closing prices of the stock. 📈")
    st.markdown("5. The app automatically selects the 'Close' column for prediction. ✔️")
    st.markdown("6. Toggle the buttons for each data preprocessing step to see its importance and status.")
    st.markdown("7. The code implementing each preprocessing step will be displayed. 💻")
    st.markdown("8. Once all preprocessing steps are completed, the app will inform you. ✅")

    get_free_space(1)

    st.subheader("🔹 Step 2: Model Training")
    st.markdown("1. In the navigation bar, click on 'Model Training'.")
    st.markdown("2. The app will load the training and testing data (X_train, X_test, y_train, y_test).")
    st.markdown("3. The model structure will be displayed. 🧠")
    st.markdown("4. Click on the 'Model Train' button to start the training process.")
    st.markdown("5. The app will show graphs illustrating the loss reduction over time. 📉")
    st.markdown("6. It will also display performance graphs for training and testing. 📈")
    st.markdown("7. Once the model is trained, you can download it using the provided link. 📥")

    get_free_space(1)

    st.subheader("🔹 Step 3: Predict")
    st.markdown("1. In the navigation bar, click on 'Predict'.")
    st.markdown("2. Import the pre-trained model using the provided option.")
    st.markdown("3. The model architecture will be displayed again. 🧠")
    st.markdown("4. Specify the number of future days you want the predictions for (1-1825).")
    st.markdown("5. Click the 'Predict' button to generate the future stock price predictions. ⏭️")
    
    get_free_space(1)

    st.markdown('## Incase you have the Model Pre-Trained and ready to use for a particular stock-code')
    st.markdown('1. First Go to the Load and preprocess Data page, and data prerprocess for the stock-code')
    st.markdown('2. Now you can directly go to the Predict page, import your model and make predictions')

    get_free_space(2)
    st.success("🎉 Congratulations! You have successfully learned how to use the Stock Prediction and Forecasting App.")
    st.balloons()


if selected == 'Load & preprocess Data':
    st.title('Extracting stock data and Preprocessing it.')
    get_free_space(1)
    stock_code = st.selectbox('Please select the stock you want', stock_codes)
    if st.button('Extract Data from Tiingo'):
        try:
            data_extraction_pipeline = DataIngestion(stock_code=stock_code, api_key=st.secrets['secrets']['auth_token'])
        except:
            data_extraction_pipeline = DataIngestion(stock_code=stock_code, api_key=st.secrets['auth_token'])

        df = data_extraction_pipeline.get_data_set()
        st.success('Dataframe successfully extracted from Tiingo')
        st.dataframe(df)
        
        # Data Visualization of Open and Close Price
        get_free_space(2)
        st.text('Close')
        st.line_chart(df['close'])
        st.area_chart(df['close'])
        st.bar_chart(df['close'])

        st.info('Close is selected as the Predict Column')
        get_free_space(2)


        # Data Preprocessing Phase
        st.header('Data Preprocessing Stage')
        get_free_space(2)
        data_preprocessor = DataPreprocessing(data=df) # Creating the Data_preprocessor Object from DataPrerprocessing Class

        with st.expander('Feature Scaling the Close Column     Phase: 1 / 4'):
            get_free_space(1)
            st.text('What is Feature Scaling and Why is it Performed')
            st.divider()
            paragraph = '''Performing feature scaling, such as normalization or standardization, is important for the "close" column in a Stock DATA, to ensure balanced and meaningful comparisons between different stock prices. Scaling helps prevent the domination of certain features due to their larger magnitudes, enhances the stability of machine learning models, and enables better convergence during training. It also ensures that the predictive models treat each feature equally, leading to more accurate and reliable stock predictions.'''
            st.write(f"<p style='white-space: pre-line;'>{paragraph}</p>", unsafe_allow_html=True)
            st.divider()
            close_df = data_preprocessor.min_max_scaling()

            np.save(f'./Numpy Arrays/{stock_code}_close_df.npy', close_df)
            
            st.success('Feature Scaling - Sucessful !!')
            get_free_space(1)
            st.text('Code :')
            feature_scale_code = '''
            from sklearn.preprocessing import StandardScaler
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            close_df = scaler.fit_transform(np.array(close_column).reshape(-1, 1))'''
            st.code(feature_scale_code)
        get_free_space(1)

        with st.expander('Splitting Close Column into Training and Testing Parts Phase: 2 / 4'):
            st.text('Why is Splitting into Training and testing')
            para = '''Its really important to Split into Training and Testing, so that we can test our model on a dataset that is never seen by the model before (Test Dataset), basically we validate our model on the Test Data, to check if its not overfitting'''
            st.write(f"<p style='white-space: pre-line;'>{para}</p>", unsafe_allow_html=True)
            st.divider()
            train_data, test_data = data_preprocessor.train_test_splitting()
            np.save(f'./Numpy Arrays/{stock_code}_train_data.npy', train_data)
            np.save(f'./Numpy Arrays/{stock_code}_test_data.npy', test_data)
            st.success('Splitting into Training and Testing - Sucessful !!')
            st.text('Code')
            splitting_code = '''
            training_size = int(len(close_df)*0.8)
            test_size = int(len(close_df)-training_size)
            train_data, test_data = close_df[0:training_size, :], close_df[training_size:len(close_df)]'''
            st.code(splitting_code)

        get_free_space(1)
        st.subheader('Converting Numpy Values to a Dataset Matrix    Phase: 3 / 4')
        st.text('What happens in this step : ')
        p = 'So, For creating each data-point in the data-frame we need the array of values and the timestep, suppose the timestep is 100, then the first data point will be the array values from [0th - 99th values in an array] is the X variable and then 100th point will be the y variable, similarly for creating the second point, we take the values from [1st - 100th ] as X variable and then 101th value as the y variable '
        st.write(f"<p style='white-space: pre-line;'>{paragraph}</p>", unsafe_allow_html=True)
        X_train, y_train =  data_preprocessor.create_dataset(dataset=train_data, timestep=150)
        X_test, y_test =  data_preprocessor.create_dataset(dataset=test_data, timestep=150)
        st.success('Converting Numpy Values to a Dataset Matrix - Sucessful !!')
        dataset_matrix_code = '''
        # Convert an array of values to dataset matrix
        def create_dataset(dataset, timestep=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-timestep-1):
            a = dataset[i: i+timestep, 0]
            dataX.append(a)
            dataY.append(dataset[i+timestep, 0])
        return np.array(dataX), np.array(dataY)

        TIME_STEPS = 150
        X_train, y_train = create_dataset(train_data, TIME_STEPS)
        X_test, y_test = create_dataset(test_data, TIME_STEPS)
        '''
        st.code(dataset_matrix_code)

        with st.expander('Reshaping to third Dimension    Phase: 4 / 5'):
            st.text('We have to reshape our X_Train and X_test to a third dimension')
            X_train, X_test = data_preprocessor.reshape_X_train_test(X_train=X_train, X_test=X_test)
            # Save files 
            np.save(f'./Numpy Arrays/{stock_code}_X_train.npy', X_train)
            np.save(f'./Numpy Arrays/{stock_code}_X_test.npy', X_test)
            np.save(f'./Numpy Arrays/{stock_code}_y_train.npy', y_train)
            np.save(f'./Numpy Arrays/{stock_code}_y_test.npy', y_test)

            st.success('Reshaped X_train and X_test to third dimension')
            st.text('Code : ')
            reshape_code = '''
            # Converting into 3-dimensional, so we can give the second and third dimension as input to the model
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            '''
            st.code(reshape_code)
        
        st.text(f'X_test shape : {X_test.shape}')
        st.text(f'X_train shape : {X_train.shape}')
        st.text(f'y_test shape : {y_test.shape}')
        st.text(f'y_train shape : {y_train.shape}')
        get_free_space(1)

        with open('current_stock_code.txt', 'w') as file:
            file.write(f'{stock_code}')

        st.success('Data Preprocessing Phase Completed, please proceed to Model Training Phase')
        st.snow()

if selected == 'Model Training':
    with open('current_stock_code.txt', 'r') as f:
        selected_stock_code = f.read().strip()

    st.title(f'Build and Train the Model for {selected_stock_code}')

    try : 
        X_train, y_train, X_test, y_test = np.load(f'./Numpy Arrays/{selected_stock_code}_X_train.npy'), np.load(f'./Numpy Arrays/{selected_stock_code}_y_train.npy'), np.load(f'./Numpy Arrays/{selected_stock_code}_X_test.npy'), np.load(f'./Numpy Arrays/{selected_stock_code}_y_test.npy') 
    
    except:
        st.warning('Please perform the Data Preprocessing Phase to extract the X_train, X_test, y_train and y_test data')
    
    else:
        st.success('Loading X_train, y_train, X_test, y_test - Successful !!')
        get_free_space(1)

        st.subheader('Model Architecture : ')
        model_architecture = """
        - Bidirectional LSTM layer with 128 units and return_sequences=True
        - Bidirectional LSTM layer with 128 units and return_sequences=True
        - Bidirectional LSTM layer with 128 units and return_sequences=True
        - Bidirectional LSTM layer with 128 units
        - Dense layer with 1 unit
        """

        # Display Model Architecture
        st.markdown(model_architecture)
        st.divider()
        get_free_space(1)
        
        st.text('Architecture Code : ')
        model_architecture = """
        from tensorflow import keras
        from keras.models import Sequential
        from keras.layers import Bidirectional, LSTM, Dense

        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(150, 1))))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(1))
        """
        get_free_space(1)
        st.code(model_architecture, language='python')

        if st.button('Compile and Train Model'):
            get_free_space(1)
            st.subheader('Model Training')
            training_pipeline = Model_Training(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            training_pipeline.create_and_train_model()
            st.info('Model has been made !!')
            training_pipeline.compile_model()
            st.info('Model has Successfully been Compiled !!')
            get_free_space(1)
            st.text('Starting Model Training')
            with st.spinner('Model is Training '):
                dnn_model, history = training_pipeline.train_model()
                st.success('Model has been Successfully Trained !!')
                st.balloons()
            
            # Saving the model
            dnn_model.save(f'./Deep Learning Models/{selected_stock_code}_bidirectional_lstm_model.h5')

            # Data Visualizations
            get_free_space(2)
            st.markdown('## Visualizing the Training Loss over each epoch during training')
            st.line_chart(history.history['loss'])
            st.area_chart(history.history['loss'])
            get_free_space(1)
            st.markdown('## Visualizing the Test Loss over each epoch during training')
            st.line_chart(history.history['val_loss'])
            st.area_chart(history.history['val_loss'])
            get_free_space(2)

            st.info('Predicting on Test Data')

            train_predict, test_predict, train_rmse, test_rmse = training_pipeline.predict_test_data()
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
            close_df = np.load(f'./Numpy Arrays/{selected_stock_code}_close_df.npy')

            para = f'''
                    Train RMSE : {train_rmse}
                    Test RMSE : {test_rmse}
                    '''
            st.write(para)
            st.markdown('## Plotting the Test Data')

            look_back = 150
            trainPredict_plot = np.empty_like(close_df)
            trainPredict_plot[:, :] = np.nan
            trainPredict_plot[look_back : len(train_predict) + look_back , :] = train_predict

            testPredict_plot = np.empty_like(close_df)
            testPredict_plot[:, :] = np.nan
            testPredict_plot[len(train_predict) + (look_back*2) + 1 : len(close_df) - 1, :] = test_predict

            fig, ax = plt.subplots()
            ax.plot(scaler.inverse_transform(close_df))
            ax.plot(trainPredict_plot)
            ax.plot(testPredict_plot)
            ax.legend()
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_title('Plotting the Output of the Train and Test data')
            st.pyplot(fig)

            st.info('Downloading Model')
            st.download_button(
                label='Download Deep Bidirectional LSTM Model',
                data = f'./Deep Learning Models/{selected_stock_code}_bidirectional_lstm_model.h5',
                file_name = f'{selected_stock_code}_bidirectional_lstm_model.h5',
                mime='application/octet-stream',
                use_container_width=True
            )


if selected == 'Predict':
    with open('current_stock_code.txt', 'r') as f:
        selected_stock_code = f.read().strip()
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    try :
       train_data, test_data = np.load(f'./Numpy Arrays/{selected_stock_code}_train_data.npy'), np.load(f'./Numpy Arrays/{selected_stock_code}_test_data.npy')
       X_train, y_train, X_test, y_test = np.load(f'./Numpy Arrays/{selected_stock_code}_X_train.npy'), np.load(f'./Numpy Arrays/{selected_stock_code}_y_train.npy'), np.load(f'./Numpy Arrays/{selected_stock_code}_X_test.npy'), np.load(f'./Numpy Arrays/{selected_stock_code}_y_test.npy') 
       stock_close_df = np.load(f'./Numpy Arrays/{selected_stock_code}_close_df.npy')
       st.success('Successfully Loaded the Train Data, Test Data, X_train, y_train, X_test and y_test')
    except:
        st.warning('Please perform the Data Preprocessing Phase to extract the X_train, X_test, y_train and y_test data')
        get_free_space(2)

    else:            
        # Load the model
        st.header('Load and Perform Predictions : ')
        get_free_space(2)

        st.subheader('Upload your .h5 model : ')
        uploaded_file = st.file_uploader("Choose a .h5 file", type="h5")
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(uploaded_file.getvalue())
                model = load_model(temp.name)
                st.success('Model is Successfully Loaded !!')
            get_free_space(2)
            st.subheader('Model Architecture')
            summary_to_text(model)
            get_free_space(2)
            st.markdown('## Please choose the number of days in the future do you want the prediction for 😄')
            choice = st.number_input('Choose the Number of Future Days that you want the output for : ', 1, 500)
            if st.button('Get Model Predictions'):
                st.info(f'You are now going to get the Future {choice} days of close values for the stock {selected_stock_code}')

                predictor_pipeline = ModelFuturePredictor(model=model, train=train_data, test=test_data, close_df=stock_close_df, num_days=choice)
                future_prediction = predictor_pipeline.predict_FuturePrice()

                day_new = np.arange(1, 151)
                day_pred = np.arange(151, 151+choice)

                # Plot the input values
                plt.figure()
                plt.plot(day_new, scaler.inverse_transform(stock_close_df[stock_close_df.shape[0]-150:]), label='Actual Data')
                plt.plot(day_pred, future_prediction, label=f'{choice} days Future Prediction')
                plt.legend()
                st.pyplot(plt)
                
                st.markdown('## Future Predictions in Table format')
                table = predictor_pipeline.show_table_future_predictions(lst_output=future_prediction, stock_code=selected_stock_code)
                st.table(table)