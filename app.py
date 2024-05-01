import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker_symbol, from_date, to_date):
    stock_data = yf.download(ticker_symbol, start=from_date, end=to_date)
    df = stock_data[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return df

# Function to train Prophet model
def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

# Function to make the forecast
def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# GUI
tabs = st.tabs(["Home", "Feed data", "Forecast", "Visualizations", "Metrics"])

with tabs[0]:
    st.header("Welcome!!")
    st.sidebar.info('Welcome!! This is a ML based stock forecast system, Click the "Need help?" button for usage instructions')
    st.title('THE STOCK MARKET PREDICTOR')
    st.markdown("""
    ## Welcome to the Stock Forecasting App!
    This app implements Machine Learning to forecast the stock values by training on historical data. 
    The model is designed for analyzing time series data with strong seasonal effects and several seasons of historical data.
    """)
    need_help = st.button('Need help?')
    help_message = "Welcome to the app! In order to use the model, fill out the input fields (From Date and To Date), enter the ticker symbol for the specific stock and define the forecast horizon. That's it!! now you are good to go!!"
    if need_help:
        st.write(help_message)

with tabs[1]:
    st.header("Feed data")
    st.write("Welcome to the data feeding section! This is where you can input the necessary details to train the forecasting model.")
    st.write("Please provide the following information:")
    st.write("1. *Ticker Symbol*: Enter the ticker symbol of the stock you want to forecast.")
    st.write("2. *Date Range*: Select the start and end dates for the historical data.")
    st.write("3. *Forecast Duration*: Set the number of days for which you want to forecast future stock prices.")
    st.write("Once you've entered the required details, click the 'Forecast Stock Prices' button below to initiate the forecasting process.")

    # Set up the layout
    st.sidebar.header('User Input Parameters')
    ticker_symbol = st.sidebar.text_input('Enter Ticker Symbol', 'RACE')
    from_date = st.sidebar.date_input('From Date', value=pd.to_datetime('2015-01-01'))
    to_date = st.sidebar.date_input('To Date', value=pd.to_datetime('today'))
    forecast_days = st.sidebar.slider('Forecast Duration (days)', min_value=1, max_value=365, value=30)
    num_forecast_days = st.sidebar.number_input('Forecast Days', min_value=1, max_value=365, value=30)

with tabs[2]:
    st.header("Forecast Results")
    st.write("Here is the forecasted data:")

    if st.sidebar.button('Forecast Stock Prices'):
        with st.spinner('Fetching data...'):
            df = fetch_stock_data(ticker_symbol, from_date, to_date)

        with st.spinner('Training model...'):
            model = train_prophet_model(df)
            forecast = make_forecast(model, forecast_days)

        st.subheader('Forecasted Data')
        st.write('The table below shows the actual and forecasted stock prices along with the lower and upper bounds of the predictions.')

        # Merge actual prices with forecasted data
        forecast_with_actual = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')

        # Display scrollable table
        st.table(forecast_with_actual[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds'))

with tabs[3]:
    st.header("Visualizations tab")
    st.write("These are the forecast visualizations")
    st.subheader('Forecast Plot')
    st.write('The plot below visualizes the predicted stock prices with their confidence intervals.')

    if 'forecast' in globals():
        fig1 = plot_plotly(model, forecast)
        fig1.update_traces(marker=dict(color='red'), line=dict(color='white'))
        st.plotly_chart(fig1)
    else:
        st.write("Please make a forecast in the 'Forecast' tab to visualize the results.")

    st.subheader('Forecast Components')
    st.write('This plot breaks down the forecast into trend, weekly, and yearly components.')

    if 'forecast' in globals():
        fig2 = plot_components_plotly(model, forecast)
        fig2.update_traces(line=dict(color='white'))
        st.plotly_chart(fig2)
    else:
        st.write("Please make a forecast in the 'Forecast' tab to visualize the results.")

with tabs[4]:
    st.header("Metrics tab")
    st.write("The metrics below provide a quantitative measure of the model’s accuracy. The Mean Absolute Error (MAE) is the average absolute difference between predicted and actual values, Mean Squared Error (MSE) is the average squared difference, and Root Mean Squared Error (RMSE) is the square root of MSE, which is more interpretable in the same units as the target variable.")

    if 'forecast' in globals():
        # Function to calculate performance metrics
        def calculate_performance_metrics(actual, predicted):
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

        actual = df['y']
        predicted = forecast['yhat'][:len(df)]
        metrics = calculate_performance_metrics(actual, predicted)
        st.metric(label="Mean Absolute Error (MAE)", value="{:.2f}".format(metrics['MAE']), delta="Lower is better")
        st.metric(label="Mean Squared Error (MSE)", value="{:.2f}".format(metrics['MSE']), delta="Lower is better")
        st.metric(label="Root Mean Squared Error (RMSE)", value="{:.2f}".format(metrics['RMSE']), delta="Lower is better")
    else:
        st.write("Please make a forecast in the 'Forecast' tab to calculate metrics.")