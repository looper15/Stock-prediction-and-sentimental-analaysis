import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import yfinance as yf
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pytz
import plotly.figure_factory as ff
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly


# Define a function to show the dropdown options for the selected country
def show_country_dropdown(country):
    st.write(f"You selected {country}")
    dropdown_options = [stock["StockName"] for stock in stocks_dict[country]]
    select_stock_name = st.selectbox("Select a Stock", dropdown_options)
    selected_stock = next((stock["StockSymbol"] for stock in stocks_dict[country] if stock["StockName"] == select_stock_name), None)
    return selected_stock

# Define the radio buttons for the countries
selected_country = st.sidebar.radio("Select a country", list(stocks_dict.keys()))

# Call the function to show the dropdown options for the selected country
selected_stock = show_country_dropdown(selected_country)



# Retrieve the historical stock data from Yahoo Finance
stock_symbol = selected_stock
start_date = datetime.now(pytz.utc) - timedelta(days=365*10) # 10 years ago
end_date = datetime.now(pytz.utc)
stock_data = si.get_data(stock_symbol, start_date, end_date)


# Retrieve the stock name 
stock_name = next((stock["StockName"] for stocks in stocks_dict.values() for stock in stocks if stock["StockSymbol"] == selected_stock), None)
print(stock_name)


# Plot the stock prices
stock_chart = plt.figure(figsize=(12, 8))
plt.plot(stock_data.index, stock_data[f'adjclose'])
plt.title(f'{stock_name} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
#plt.show()
st.header(f'Price graph of {stock_name} from 2013 to June 2023')
st.pyplot(stock_chart)


#Plot Volume Graph
figVolume = plt.figure(figsize=(15, 8))
# Create a scatter plot
plt.scatter(stock_data.index, stock_data['volume'], s=5, c='red')

# Add labels and title
plt.title(f'{stock_name} Stock volume')
plt.xlabel('Date')
plt.ylabel('volume in 10^6')

# Show the plot
plt.show()



# Prepare the data for the EMA calculations
periods = [50, 100, 200]
for period in periods:
    stock_data[f'{period}d_EMA'] = stock_data['adjclose'].ewm(span=period, adjust=False).mean()


    # Plot the stock prices and EMA
fig1=plt.figure(figsize=(12, 8))
plt.plot(stock_data.index, stock_data['adjclose'], label='Actual')
for period in periods:
    plt.plot(stock_data.index, stock_data[f'{period}d_EMA'], label=f'{period}d EMA')
plt.title(f'{stock_name} Stock Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()
st.text("\n")
st.header(f'EMA graph for {stock_name}')
st.pyplot(fig1)


# Retrieve news articles for the same time period
rss_feed = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_symbol}&region=US&lang=en-US'
news_articles = []
feed = feedparser.parse(rss_feed)
for entry in feed.entries:
    if start_date <= datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z') <= end_date:
        news_articles.append(entry.title)


        # Perform sentiment analysis using TextBlob
sentiment_scores = []
for article in news_articles:
    blob = TextBlob(article)
    sentiment_scores.append(blob.sentiment.polarity)


plt.show()
fig3=plt.figure(figsize=(12, 8))

# Plot the histogram of yearly returns (with negatives converted to positives)
plt.subplot(212)
pos_returns = returns.apply(lambda x: abs(x))
plt.hist(pos_returns, bins=10)
plt.title(f'{stock_name} Sentimental analysis Histogram')
plt.xlabel('Return')
plt.ylabel('Frequency')

plt.show()
st.text("\n")
st.pyplot(fig3)

stock_data.reset_index(inplace=True)
stock_data['Date'] = stock_data['index']


# Prepare the data for the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['adjclose'].values.reshape(-1, 1))

prediction_days = 30
x_train, y_train = [], []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i-prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=1, batch_size=1)



# Use Prophet to predict the future prices
prophet_df = stock_data.reset_index()[['Date', 'adjclose']].rename({'Date': 'ds', 'adjclose': 'y'}, axis=1)
m = Prophet(daily_seasonality=True)
m.fit(prophet_df)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plot the results using Plotly
fig = plot_plotly(m, forecast)
fig.update_layout(title=f'{stock_name} Stock Price Prediction',
                  xaxis_title='Date',
                  yaxis_title='Adjusted Close Price')


# fig.show()
final = plot_plotly(m, forecast, uncertainty=True, xlabel="Date", ylabel="Price")