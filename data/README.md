# Data Directory

This directory contains data processing modules and sample data structures.

## Directory Structure

```
data/
├── alpha_vantage_cache/     # Alpha Vantage stock data cache (excluded from git)
├── data_cache/              # Processed data cache (excluded from git)
├── news_cache/              # News data cache (excluded from git)
├── polygon_news_data/       # Polygon.io news data (excluded from git)
└── README.md               # This file
```

## Data Sources

### Stock Data (Alpha Vantage)
- **Format**: OHLC (Open, High, Low, Close) data
- **Frequency**: Daily
- **Tickers**: S&P 500 companies
- **Cache**: Stored as pickle files for fast access

### News Data (Polygon.io)
- **Format**: JSON with sentiment scores
- **Content**: Financial news articles
- **Sentiment**: Pre-calculated sentiment scores
- **Cache**: Organized by ticker symbol

## Usage

The data directories are automatically created when you run the training scripts. The cache directories help speed up subsequent runs by avoiding redundant API calls.

## Privacy Note

All data directories are excluded from version control via `.gitignore` to protect API usage and ensure users download their own data.
