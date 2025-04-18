# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from llm_model import get_stock_data, analyze_news_sentiment, train_stock_price_model, predict_stock_price, visualize_predictions

app = Flask(__name__)
initialized = False
stock_price_model = None

@app.before_request
def initialize_model():
    global initialized, stock_price_model
    if not initialized:
        stock_data = get_stock_data('AAPL')
        news_sentiment = analyze_news_sentiment('Apple')

        if stock_data.empty or news_sentiment.empty:
            print("Initialization failed: Insufficient data for training the model.")
            return

        stock_price_model = train_stock_price_model(stock_data, news_sentiment)
        initialized = True

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.json.get('symbol', '').upper()
    stock_data = get_stock_data(symbol)
    news_sentiment = analyze_news_sentiment(symbol)

    if stock_data.empty or news_sentiment.empty:
        return jsonify({'error': 'Insufficient data for prediction'}), 400

    try:
        predictions = predict_stock_price(stock_price_model, stock_data, news_sentiment)
        return jsonify(predictions.to_dict('records'))
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == "__main__":
    app.run(debug=True)
