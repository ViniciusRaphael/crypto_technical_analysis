# Cryptocurrency Data Scraping and Analysis

This project involves scraping cryptocurrency OHLCV (Open, High, Low, Close, Volume) data from Binance and Kucoin exchanges and performing technical analysis by calculating various indicators such as ADX (Average Directional Index), Ichimoku Cloud, and MACD (Moving Average Convergence Divergence).

## Overview

The project consists of the following key components:

- **Data Scraping**: Utilizing Binance and Kucoin APIs to retrieve OHLCV data for various cryptocurrencies.
- **Technical Indicators Calculation**:
  - ADX: Calculating the Average Directional Index to measure the strength of a trend in the price movement of cryptocurrencies.
  - Ichimoku Cloud: Generating Ichimoku Cloud lines for analyzing support, resistance, and trend direction.
  - MACD: Computing the Moving Average Convergence Divergence for identifying potential buy or sell signals.

## Folders Structure

    ├── api/ # Main files
    ├── src/ # Main project source code
    │ ├── notebooks/ # Jupyter notebooks for experimentation and data exploration
    │ └── utils/ # Utility functions and helpers
    ├── data/ # saved datasets (.parquet)
    ├── output/ # Output from the scr folders (trained models, backtesting, predictions, etc)
    │ ├── accuracy/ # Model accuracy
    │ ├── backtest/ # Backtesting models predictions
    │ ├── models/ # Trained models
    │ ├── predict/ # Predictions for trained model
    │ ├── simulations/ # Backtesting models predictions
    ├── .gitignore # Gitignore file to avoid versioning unnecessary files
    ├── README.md # Project instructions and documentation
    ├── setup.py # Configs for run the libs and class from the project
    └── requirements.txt # Project Python dependencies
