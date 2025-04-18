# -*- coding: utf-8 -*-
import os
import yfinance as yf
import requests
import pandas as pd
from transformers import pipeline
import praw
from tenacity import retry, stop_after_attempt, wait_exponential
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify

# ✅ Load environment variables
load_dotenv()

# ✅ Secure API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
HF_API_KEY = os.getenv("HF_API_KEY")

# ✅ FinBERT Setup
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# ✅ Fetch stock data
def get_stock_data(symbol: str, years=3):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=years)
    stock_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_data.reset_index(inplace=True)
    return stock_data

# ✅ Fetch news
def fetch_news(company):
    # Example: Replace this with your actual news fetching logic
    return [
        {'date': '2025-04-17', 'title': f'{company} stock rises on positive earnings'},
        {'date': '2025-04-16', 'title': f'{company} faces challenges in the market'},
    ]

# ✅ News Sentiment Analysis
def analyze_news_sentiment(company):
    # Fetch news headlines
    headlines = fetch_news(company)
    if not headlines:
        return pd.DataFrame()  # Return an empty DataFrame if no headlines are found

    sentiments = [
        {
            'date': headline['date'],  # Ensure 'date' is included
            'headline': headline['title'],
            'sentiment': finbert(headline['title'])[0]['label'],
            'confidence': finbert(headline['title'])[0]['score']
        }
        for headline in headlines
    ]
    return pd.DataFrame(sentiments)  # Return as DataFrame

# ✅ Reddit Sentiment Analysis
@retry(stop=stop_after_attempt(2))
def analyze_reddit_sentiment(subreddit="stocks", limit=50):
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_SECRET, user_agent="stock-analysis/1.0")
    posts = reddit.subreddit(subreddit).hot(limit=limit)
    return pd.DataFrame([
        {
            'title': post.title,
            'score': post.score,
            'sentiment': finbert(post.title)[0]['label'],
            'confidence': finbert(post.title)[0]['score'],
            'created_utc': pd.to_datetime(post.created_utc, unit='s')
        }
        for post in posts
    ])

# ✅ Visualization
def visualize_results(news_df, reddit_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=news_df['date'], y=news_df['confidence'], mode='markers', name='News Sentiment'))
    fig.add_trace(go.Scatter(x=reddit_df['created_utc'], y=reddit_df['score'], mode='lines+markers', name='Reddit Engagement'))
    fig.update_layout(title='Sentiment Analysis', template='plotly_dark')
    fig.show()

# ✅ Train a Stock Price Prediction Model
def train_stock_price_model(stock_data, sentiment_data):
    # Ensure sentiment_data is a DataFrame
    if sentiment_data.empty:
        raise ValueError("Sentiment data is empty")

    # Merge stock data and sentiment data on date
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
    combined_data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date', how='inner')

    # Prepare features and target
    combined_data['Sentiment_Score'] = combined_data['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    X = combined_data[['Open', 'High', 'Low', 'Volume', 'Sentiment_Score']]
    y = combined_data['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"Model Mean Squared Error: {mse}")

    return model

# ✅ Predict Stock Prices
def predict_stock_price(model, stock_data, sentiment_data):
    if stock_data.empty or sentiment_data.empty:
        raise ValueError("Insufficient data for prediction")

    # Merge stock data and sentiment data on date
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
    combined_data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date', how='inner')

    # Prepare features
    combined_data['Sentiment_Score'] = combined_data['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    X = combined_data[['Open', 'High', 'Low', 'Volume', 'Sentiment_Score']]
    combined_data['Predicted_Close'] = model.predict(X)
    return combined_data[['Date', 'Close', 'Predicted_Close']]

# ✅ Visualize Predictions
def visualize_predictions(predictions_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Close'], mode='lines+markers', name='Actual Close'))
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted_Close'], mode='lines+markers', name='Predicted Close'))
    fig.update_layout(title='Stock Price Prediction', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    fig.show()

# ✅ Flask App Initialization
app = Flask(__name__)
initialized = False
stock_price_model = None

@app.before_request
def initialize_model():
    global initialized
    if not initialized:
        global stock_price_model
        stock_data = get_stock_data('AAPL')  # Example stock symbol
        news_sentiment = analyze_news_sentiment('Apple')

        if stock_data is None or news_sentiment.empty:
            print("Initialization failed: Insufficient data for training the model.")
            initialized = False
            return

        stock_price_model = train_stock_price_model(stock_data, news_sentiment)
        initialized = True

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.json.get('symbol', '').upper()
    stock_data = get_stock_data(symbol)
    news_sentiment = analyze_news_sentiment(symbol)

    if stock_data is None or news_sentiment.empty:
        return jsonify({'error': 'Insufficient data for prediction'}), 400

    try:
        predictions = predict_stock_price(stock_price_model, stock_data, news_sentiment)
        return jsonify(predictions.to_dict('records'))
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == "__main__":
    stock_data = get_stock_data('AAPL')
    news_results = analyze_news_sentiment('Apple')

    if not news_results.empty:
        stock_price_model = train_stock_price_model(stock_data, news_results)
        predictions = predict_stock_price(stock_price_model, stock_data, news_results)
        visualize_predictions(predictions)
