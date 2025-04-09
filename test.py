#!/usr/bin/env python
"""
Bitcoin price forecasting test script with TensorFlow fixes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import sys
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("__main__")


def create_price_forecast_chart(price_data, forecast, ticker="BTC-USD"):
    """Create a price chart with forecast."""
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(price_data.index[-30:], price_data['Close'][-30:], label='Historical Price', color='blue')

    # Extract forecast data
    dates = []
    prices = forecast.get('forecast', [])

    if not prices:
        logger.error("No forecast data available")
        return None

    # Create dates for forecast
    last_date = price_data.index[-1]
    for i in range(len(prices)):
        dates.append(last_date + timedelta(days=i + 1))

    # Plot forecast
    plt.plot(dates, prices, label='Forecast', color='red', linestyle='--', marker='o')

    # Add labels and title
    plt.title(f"{ticker} Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save chart
    output_file = f"{ticker}_forecast.png"
    plt.savefig(output_file)
    plt.close()

    return output_file


# --- Modified PriceForecastingEngine with fixed methods ---

class FixedPriceForecastingEngine:
    """Fixed price forecasting engine."""

    def __init__(self):
        """Initialize the engine."""
        self.models = {}
        self.scalers = {}

        # Import necessary libraries inside the class to avoid import issues
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler

        self.tf = tf
        self.Sequential = Sequential
        self.LSTM = LSTM
        self.Dense = Dense
        self.Dropout = Dropout
        self.MinMaxScaler = MinMaxScaler

    def prepare_data(self, price_data, look_back=60):
        """Prepare data for LSTM model."""
        try:
            # Check if we have 'Close' column
            if 'Close' not in price_data.columns:
                logger.error("Price data missing 'Close' column")
                return None, None, None

            # Check if we have enough data
            if len(price_data) <= look_back:
                logger.error(f"Not enough data points. Need more than {look_back} points.")
                return None, None, None

            # Extract close prices
            close_prices = price_data['Close'].values.reshape(-1, 1)

            # Create and fit scaler
            scaler = self.MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - look_back):
                X.append(scaled_data[i:i + look_back, 0])
                y.append(scaled_data[i + look_back, 0])

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)

            # Check if we have data
            if len(X) == 0 or len(y) == 0:
                logger.error("Failed to create input sequences")
                return None, None, None

            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)

            return X, y, scaler

        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            return None, None, None

    def train_model(self, ticker, price_data, epochs=50, look_back=60):
        """Train LSTM model for price forecasting."""
        try:
            logger.info(f"Training forecast model for {ticker}")

            # Prepare data
            X, y, scaler = self.prepare_data(price_data, look_back)
            if X is None or y is None or scaler is None:
                return {'error': 'Failed to prepare data'}

            # Split into train/test
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Create model
            model = self.Sequential()
            model.add(self.LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
            model.add(self.Dropout(0.2))
            model.add(self.LSTM(units=50))
            model.add(self.Dropout(0.2))
            model.add(self.Dense(units=1))

            # Compile model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train model
            batch_size = min(32, len(X_train))
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0
            )

            # Store model and scaler
            self.models[ticker] = model
            self.scalers[ticker] = scaler

            val_loss = history.history['val_loss'][-1]
            logger.info(f"Model trained for {ticker} with val_loss: {val_loss:.6f}")

            return {
                'ticker': ticker,
                'epochs': epochs,
                'val_loss': val_loss
            }

        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
            return {'error': str(e)}

    def forecast_prices(self, ticker, price_data, forecast_days=7):
        """Forecast future prices."""
        try:
            # Check if model exists
            if ticker not in self.models:
                logger.info(f"No model found for {ticker}, training now")
                result = self.train_model(ticker, price_data)
                if 'error' in result:
                    return {
                        'error': result['error'],
                        'forecast': [],
                        'forecast_dates': []
                    }

            model = self.models[ticker]
            scaler = self.scalers[ticker]

            # Get last sequence of data
            look_back = 60
            last_sequence = price_data['Close'].values[-look_back:].reshape(-1, 1)
            last_sequence_scaled = scaler.transform(last_sequence)

            # Flatten the sequence
            current_sequence = last_sequence_scaled.flatten()

            # Generate forecasts
            forecasted_values = []

            for _ in range(forecast_days):
                # Prepare input shape: [1, look_back, 1]
                x_input = current_sequence.reshape(1, look_back, 1)

                # Predict next value
                next_pred = model.predict(x_input, verbose=0)
                next_val = next_pred[0, 0]

                # Add to forecast
                forecasted_values.append(next_val)

                # Update sequence (remove first, add new prediction)
                current_sequence = np.append(current_sequence[1:], next_val)

            # Convert back to original scale
            forecasted_array = np.array(forecasted_values).reshape(-1, 1)
            forecasted_prices = scaler.inverse_transform(forecasted_array)

            # Create forecast dates
            last_date = price_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

            return {
                'ticker': ticker,
                'last_known_price': float(price_data['Close'].iloc[-1]),
                'forecast': [float(val[0]) for val in forecasted_prices],
                'forecast_dates': [str(date) for date in forecast_dates]
            }

        except Exception as e:
            logger.error(f"Error in forecast_prices: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            return {
                'error': str(e),
                'forecast': [],
                'forecast_dates': []
            }


def main():
    """Main function."""
    try:
        # Download Bitcoin price data
        print("Downloading Bitcoin price data...")
        ticker = "BTC-USD"
        days = 100  # More data for better training

        price_data = yf.download(ticker, period=f"{days}d")
        print(f"Downloaded {len(price_data)} days of price data for {ticker}")

        if price_data.empty or len(price_data) < 60:
            logger.error("Not enough price data for forecasting")
            return

        # Initialize forecasting engine
        print("Initializing forecasting engine...")
        forecasting_engine = FixedPriceForecastingEngine()

        # Generate forecast
        print(f"Generating 7-day forecast for {ticker}...")
        forecast = forecasting_engine.forecast_prices(
            ticker=ticker,
            price_data=price_data,
            forecast_days=7
        )

        if 'error' in forecast and forecast['error']:
            logger.error(f"Forecast error: {forecast['error']}")
            return

        # Display forecast
        print("\nFORECAST RESULTS:")
        print(f"Last known price: ${forecast['last_known_price']:.2f}")
        print("Forecasted prices:")

        # Check if we have forecast data
        if 'forecast' in forecast and forecast['forecast']:
            for i, (date, price) in enumerate(zip(forecast['forecast_dates'], forecast['forecast'])):
                print(f"  Day {i + 1} ({date}): ${price:.2f}")

            # Create chart
            print("\nCreating visualization...")
            output_file = create_price_forecast_chart(price_data, forecast, ticker)

            if output_file:
                print(f"Visualization saved to: {os.path.abspath(output_file)}")
        else:
            print("No forecast data available")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()