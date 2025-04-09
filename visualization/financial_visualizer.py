"""
Visualization component for FinSight system.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from io import BytesIO
import base64
import logging

logger = logging.getLogger(__name__)


class FinancialVisualizer:
    """Financial data visualization engine."""

    def __init__(self, theme='dark_background'):
        """Initialize the visualizer."""
        plt.style.use(theme)
        self.fig_size = (10, 6)

    def _convert_to_base64(self, fig, save_path=None):
        """
        Convert matplotlib figure to base64.

        Args:
            fig: Matplotlib figure
            save_path: Optional path to save the figure as an image file

        Returns:
            Base64 encoded image
        """
        # Save figure to file if path is provided
        if save_path:
            fig.savefig(save_path, format='png', bbox_inches='tight')
            logger.info(f"Figure saved to: {save_path}")

        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)

        return base64.b64encode(image_png).decode('utf-8')

    def create_price_chart(self, price_data, ticker, forecast=None):
        """
        Create a price chart with optional forecast.

        Args:
            price_data: DataFrame with price data
            ticker: Ticker symbol
            forecast: Optional forecast data

        Returns:
            Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=self.fig_size)

        # Plot historical prices
        ax.plot(price_data.index, price_data['Close'], label='Historical Price', color='#2196F3', linewidth=2)

        # Add forecast if provided
        if forecast:
            # Parse dates and values
            dates = pd.to_datetime(forecast['forecast_dates'])
            values = forecast['forecast']

            # Plot forecast
            ax.plot(dates, values,
                    label='Price Forecast',
                    color='#FF9800',
                    linestyle='--',
                    linewidth=2,
                    marker='o')

            # Add shaded area for uncertainty
            if 'upper_bound' in forecast and 'lower_bound' in forecast:
                ax.fill_between(
                    dates,
                    forecast['lower_bound'],
                    forecast['upper_bound'],
                    color='#FF9800',
                    alpha=0.2,
                    label='Forecast Uncertainty'
                )

        # Set title and labels
        ax.set_title(f'{ticker} Price Chart', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Format y-axis
        ax.yaxis.set_major_formatter('${x:,.2f}')

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)

        # Tight layout
        fig.tight_layout()

        save_path = f"{ticker}_price_chart.png"
        base64_img = self._convert_to_base64(fig, save_path=save_path)
        return base64_img,save_path

    def create_sentiment_chart(self, sentiment_data):
        """
        Create a sentiment analysis visualization.

        Args:
            sentiment_data: Sentiment analysis data

        Returns:
            Base64 encoded image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Pie chart of overall sentiment
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [
            sentiment_data.get('positive_count', 0),
            sentiment_data.get('negative_count', 0),
            sentiment_data.get('neutral_count', 0)
        ]
        colors = ['#4CAF50', '#F44336', '#2196F3']
        explode = (0.1, 0.1, 0.1)

        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        ax1.axis('equal')
        ax1.set_title('Overall Sentiment Distribution', fontsize=14)

        # Bar chart of key factors
        if 'key_factors' in sentiment_data:
            pos_factors = sentiment_data['key_factors'].get('positive', [])
            neg_factors = sentiment_data['key_factors'].get('negative', [])

            factors = []
            values = []
            colors = []

            for factor in pos_factors[:3]:
                factors.append(factor)
                values.append(1)
                colors.append('#4CAF50')

            for factor in neg_factors[:3]:
                factors.append(factor)
                values.append(-1)
                colors.append('#F44336')

            y_pos = np.arange(len(factors))
            ax2.barh(y_pos, values, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(factors)
            ax2.set_xlabel('Sentiment Impact')
            ax2.set_title('Key Sentiment Factors', fontsize=14)
            ax2.set_xlim(-1.5, 1.5)

        # Tight layout
        fig.tight_layout()

        save_path = "sentiment_chart.png"
        base64_img = self._convert_to_base64(fig, save_path=save_path)
        return base64_img, save_path

    def create_technical_dashboard(self, price_data, ticker, technical_analysis=None):
        """
        Create a technical analysis dashboard.

        Args:
            price_data: DataFrame with price data
            ticker: Ticker symbol
            technical_analysis: Optional technical analysis data

        Returns:
            Base64 encoded image
        """
        fig = plt.figure(figsize=(12, 10))

        # Create grid for subplots
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

        # Price chart with MA
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(price_data.index, price_data['Close'], label='Price', color='#2196F3')

        # Add moving averages if we have enough data
        if len(price_data) >= 50:
            price_data['MA20'] = price_data['Close'].rolling(window=20).mean()
            price_data['MA50'] = price_data['Close'].rolling(window=50).mean()

            ax1.plot(price_data.index, price_data['MA20'], label='20-day MA', color='#FF9800')
            ax1.plot(price_data.index, price_data['MA50'], label='50-day MA', color='#F44336')

        ax1.set_title(f'{ticker} Price with Moving Averages', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Volume chart
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        # Check if Volume data exists
        # Volume chart with explicit type conversion
        if 'Volume' in price_data.columns:
            # Convert dates and values to Python scalars explicitly
            dates = [pd.Timestamp(x).to_pydatetime() for x in price_data.index]
            volumes = [float(v) for v in price_data['Volume'].values]

            # Create bar chart with explicit Python scalar values
            ax2.bar(dates, volumes, color='#2196F3', alpha=0.6)
        else:
            ax2.text(0.5, 0.5, "Volume data not available",
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)
        ax2.set_title('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # RSI indicator
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        # Calculate RSI
        delta = price_data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        ax3.plot(price_data.index, rsi, color='#673AB7')
        ax3.axhline(70, color='#F44336', linestyle='--', alpha=0.5)
        ax3.axhline(30, color='#4CAF50', linestyle='--', alpha=0.5)
        ax3.set_title('RSI (14)', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        # Add technical analysis annotations if provided
        if technical_analysis:
            # Add support levels
            for level in technical_analysis.get('support_levels', [])[:3]:
                ax1.axhline(level, color='#4CAF50', linestyle='--', alpha=0.7)
                ax1.text(price_data.index[-1], level, f'Support: {level}',
                         color='#4CAF50', ha='right', va='bottom')

            # Add resistance levels
            for level in technical_analysis.get('resistance_levels', [])[:3]:
                ax1.axhline(level, color='#F44336', linestyle='--', alpha=0.7)
                ax1.text(price_data.index[-1], level, f'Resistance: {level}',
                         color='#F44336', ha='right', va='top')

            # Add recommendation
            recommendation = technical_analysis.get('recommendation', 'hold').upper()
            timeframe = technical_analysis.get('recommendation_timeframe', 'medium')

            if recommendation == 'BUY':
                color = '#4CAF50'
            elif recommendation == 'SELL':
                color = '#F44336'
            else:
                color = '#FFC107'

            fig.text(0.5, 0.01, f'Recommendation: {recommendation} ({timeframe} term)',
                     color=color, ha='center', fontsize=14, weight='bold')

        # Format and layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)

        save_path = f"{ticker}_technical_chart.png"
        base64_img = self._convert_to_base64(fig, save_path=save_path)
        return base64_img, save_path