"""
Price forecasting module for FinSight system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

logger = logging.getLogger(__name__)


class PriceForecastingEngine:
    """Price forecasting engine using LSTM neural networks."""

    def __init__(self):
        """Initialize the forecasting engine."""
        self.models = {}
        self.scalers = {}

    def prepare_data(self, price_data, look_back=30):
        """
        Prepare data for LSTM model.
        """
        logger.info(
            f"prepare_data called with price_data shape: {price_data.shape if hasattr(price_data, 'shape') else 'unknown'}, look_back: {look_back}")

        try:
            # Ensure we have 'Close' column
            if 'Close' not in price_data.columns:
                logger.error("Price data missing 'Close' column")
                return None, None, None

            # Log price_data info
            logger.info(f"Price data date range: {price_data.index[0]} to {price_data.index[-1]}")
            logger.info(f"Number of data points: {len(price_data)}")
            logger.info(f"Column names: {list(price_data.columns)}")

            # Check if we have enough data
            if len(price_data) <= look_back:
                logger.error(f"Not enough data points. Need more than {look_back} points, but got {len(price_data)}")
                return None, None, None

            # Extract close prices
            close_prices = price_data['Close'].values.reshape(-1, 1)
            logger.info(f"Close prices shape: {close_prices.shape}")

            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            logger.info(f"Scaled data shape: {scaled_data.shape}")

            # Create sequences for LSTM
            X, y = [], []
            for i in range(len(scaled_data) - look_back):
                X.append(scaled_data[i:i + look_back, 0])
                y.append(scaled_data[i + look_back, 0])

            logger.info(f"Created {len(X)} sequences")

            # Check if we have sequences
            if not X or not y:
                logger.error("Failed to create sequences for LSTM")
                return None, None, None

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            logger.info(f"X shape before reshape: {X.shape}")

            # Ensure X is not empty before reshaping
            if X.size == 0:
                logger.error("Empty X array, cannot reshape")
                return None, None, None

            # Reshape for LSTM [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            logger.info(f"X shape after reshape: {X.shape}")

            return X, y, scaler

        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None

    def train_model(self, ticker, price_data, epochs=50, look_back=30):
        """
        Train LSTM model for price prediction.

        Args:
            ticker: Ticker symbol
            price_data: DataFrame with price data
            epochs: Number of training epochs
            look_back: Number of previous time steps to use

        Returns:
            Training metrics
        """
        logger.info(f"Training forecast model for {ticker}")

        if len(price_data) < look_back + 2:
            max_lookback = max(5, len(price_data) // 3)
            logger.warning(f"Adjusting look_back from {look_back} to {max_lookback} due to limited data")
            look_back = max_lookback

        # Prepare data
        X, y, scaler = self.prepare_data(price_data, look_back)
        if X is None:
            return {'error': 'Failed to prepare data'}

        # Split into train/test sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Create model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile and train
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )

        # Store model and scaler
        self.models[ticker] = model
        self.scalers[ticker] = scaler

        # Calculate validation loss
        val_loss = history.history['val_loss'][-1]
        logger.info(f"Model trained for {ticker} with validation loss: {val_loss:.6f}")

        return {
            'ticker': ticker,
            'epochs': epochs,
            'val_loss': val_loss,
            'look_back': look_back
        }

    def forecast_prices(self, ticker, price_data, forecast_days=7, look_back=30):
        """
        Forecast future prices.

        Args:
            ticker: Ticker symbol
            price_data: DataFrame with price data
            forecast_days: Number of days to forecast

        Returns:
            Dictionary with forecast results
        """
        try:
            # Check if model exists
            if ticker not in self.models:
                logger.info(f"No model found for {ticker}, training now")
                train_result = self.train_model(ticker, price_data, look_back=look_back)

                # Check if training was successful
                if isinstance(train_result, dict) and 'error' in train_result:
                    logger.error(f"Failed to train model for {ticker}: {train_result['error']}")
                    return {
                        'error': train_result['error'],
                        'ticker': ticker,
                        'forecast_generated': pd.Timestamp.now().isoformat(),
                        'last_known_price': float(price_data['Close'].iloc[-1].item()) if not price_data.empty else 0,
                        'forecast': [],
                        'forecast_dates': []
                    }

            model = self.models.get(ticker)
            scaler = self.scalers.get(ticker)

            # Check if model and scaler exist
            if model is None or scaler is None:
                logger.error(f"Model or scaler not found for {ticker}")
                return {
                    'error': 'Model or scaler not found',
                    'ticker': ticker,
                    'forecast_generated': pd.Timestamp.now().isoformat(),
                    'last_known_price': float(price_data['Close'].iloc[-1].item()) if not price_data.empty else 0,
                    'forecast': [],
                    'forecast_dates': []
                }

            # Ensure we have enough data
            look_back = look_back  # Default value, should match training
            if len(price_data) < look_back:
                logger.error(f"Not enough historical data for {ticker}. Need at least {look_back} points.")
                return {
                    'error': f'Not enough historical data. Need at least {look_back} points.',
                    'ticker': ticker,
                    'forecast_generated': pd.Timestamp.now().isoformat(),
                    'last_known_price': float(price_data['Close'].iloc[-1].item()) if not price_data.empty else 0,
                    'forecast': [],
                    'forecast_dates': []
                }

            # Get the last sequence
            last_sequence = price_data['Close'].values[-look_back:].reshape(-1, 1)
            last_sequence_scaled = scaler.transform(last_sequence)

            # Flatten the sequence
            current_sequence = last_sequence_scaled.flatten()

            # Forecast process
            forecasted_scaled = []

            for _ in range(forecast_days):
                # *** CRITICAL FIX: Prepare input with correct shape for TensorFlow ***
                # Shape should be [1, look_back, 1] - [batch_size, timesteps, features]
                x_input = current_sequence.reshape(1, look_back, 1)

                # Predict next value with fixed indexing
                next_pred = model.predict(x_input, verbose=0)
                next_val = next_pred[0, 0]

                # Add to forecast
                forecasted_scaled.append(next_val)

                # Update sequence by dropping first value and adding prediction
                current_sequence = np.append(current_sequence[1:], next_val)

            # Convert forecasted values to array for inverse transformation
            forecasted_scaled_array = np.array(forecasted_scaled).reshape(-1, 1)

            # Invert scaling to get actual price values
            forecasted_values = scaler.inverse_transform(forecasted_scaled_array)

            # Create date range for forecast
            last_date = price_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
            else:
                forecast_dates = [f"Day {i + 1}" for i in range(forecast_days)]

            # Build result object
            forecast_result = {
                'ticker': ticker,
                'forecast_generated': pd.Timestamp.now().isoformat(),
                'last_known_price': float(price_data['Close'].iloc[-1]),
                'forecast': [float(val[0]) for val in forecasted_values],
                'forecast_dates': [str(date) for date in forecast_dates]
            }

            return forecast_result

        except Exception as e:
            logger.error(f"Error in forecast_prices: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            return {
                'error': str(e),
                'ticker': ticker,
                'forecast_generated': pd.Timestamp.now().isoformat(),
                'last_known_price': float(price_data['Close'].iloc[-1].item()) if not price_data.empty else 0,
                'forecast': [],
                'forecast_dates': []
            }

    def diagnose_forecasting_process(self, ticker, price_data):
        """
        Diagnose the forecasting process for debugging.

        Args:
            ticker: Ticker symbol
            price_data: DataFrame with price data

        Returns:
            Dictionary with diagnostic information
        """
        logger.info(f"Starting forecasting diagnosis for {ticker}")

        diagnostics = {
            "ticker": ticker,
            "data_stats": {},
            "preparation": {},
            "training": {},
            "forecasting": {},
            "errors": []
        }

        try:
            # Data stats
            diagnostics["data_stats"] = {
                "total_rows": len(price_data),
                "date_range": f"{price_data.index[0]} to {price_data.index[-1]}",
                "columns": list(price_data.columns),
                "has_close_column": "Close" in price_data.columns,
                "close_min": float(price_data["Close"].min()) if "Close" in price_data.columns else None,
                "close_max": float(price_data["Close"].max()) if "Close" in price_data.columns else None,
                "close_mean": float(price_data["Close"].mean()) if "Close" in price_data.columns else None
            }

            # Data preparation
            try:
                logger.info("Testing data preparation...")
                look_back = 60

                # Try different look_back values if needed
                if len(price_data) <= look_back:
                    adjusted_look_back = max(5, len(price_data) // 3)
                    diagnostics["preparation"]["adjusted_look_back"] = adjusted_look_back
                    look_back = adjusted_look_back

                X, y, scaler = self.prepare_data(price_data, look_back)

                if X is not None and y is not None and scaler is not None:
                    diagnostics["preparation"]["success"] = True
                    diagnostics["preparation"]["X_shape"] = X.shape
                    diagnostics["preparation"]["y_shape"] = y.shape
                    diagnostics["preparation"]["num_sequences"] = len(X)
                else:
                    diagnostics["preparation"]["success"] = False
                    diagnostics["errors"].append("Data preparation failed")
            except Exception as e:
                diagnostics["preparation"]["success"] = False
                diagnostics["preparation"]["error"] = str(e)
                diagnostics["errors"].append(f"Data preparation exception: {str(e)}")

            # Skip training and forecasting if preparation failed
            if not diagnostics["preparation"].get("success", False):
                return diagnostics

            # Model training
            try:
                logger.info("Testing model training...")
                training_result = self.train_model(ticker, price_data, epochs=10, look_back=look_back)
                diagnostics["training"] = training_result

                if "error" in training_result:
                    diagnostics["errors"].append(f"Training error: {training_result['error']}")
            except Exception as e:
                diagnostics["training"]["success"] = False
                diagnostics["training"]["error"] = str(e)
                diagnostics["errors"].append(f"Training exception: {str(e)}")

            # Skip forecasting if training failed
            if "error" in diagnostics["training"]:
                return diagnostics

            # Forecasting
            try:
                logger.info("Testing forecasting...")
                forecast = self.forecast_prices(ticker, price_data, forecast_days=3)

                diagnostics["forecasting"] = {
                    "success": "error" not in forecast,
                    "last_known_price": forecast.get("last_known_price"),
                    "num_forecasted_days": len(forecast.get("forecast", [])),
                    "forecast_sample": forecast.get("forecast", [])[:3]
                }

                if "error" in forecast:
                    diagnostics["errors"].append(f"Forecasting error: {forecast['error']}")
            except Exception as e:
                diagnostics["forecasting"]["success"] = False
                diagnostics["forecasting"]["error"] = str(e)
                diagnostics["errors"].append(f"Forecasting exception: {str(e)}")

        except Exception as e:
            diagnostics["errors"].append(f"Overall diagnosis error: {str(e)}")

        # Save diagnostics to file
        try:
            import json
            with open(f"{ticker}_forecast_diagnosis.json", "w") as f:
                json.dump(diagnostics, f, indent=2, default=str)
            logger.info(f"Diagnostics saved to {ticker}_forecast_diagnosis.json")
        except:
            pass

        return diagnostics