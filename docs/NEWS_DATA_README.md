# Alpha Vantage News Data Utility

This module provides comprehensive news data functionality for the Hyper framework using the Alpha Vantage API. It includes caching, sentiment analysis, and data processing capabilities.

## Features

- **News Sentiment Data**: Download news articles with sentiment analysis
- **Caching System**: Automatic caching to reduce API calls
- **Rate Limiting**: Built-in rate limiting for Alpha Vantage API
- **Data Processing**: Clean and process news data
- **Sentiment Analysis**: Analyze sentiment distribution and statistics
- **News Summaries**: Generate summaries of top news articles

## Setup

### 1. Install Dependencies

The Alpha Vantage library is included in the requirements:

```bash
pip install alpha-vantage>=2.3.1
```

### 2. Get Alpha Vantage API Key

1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free API key
3. Set the environment variable:

```bash
export ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

### 3. API Limits

- **Free Tier**: 5 API calls per minute
- **Premium Tier**: 500 API calls per minute

The utility includes built-in rate limiting to stay within these limits.

## Usage

### Basic News Download

```python
from src.data.news_data import get_news_sentiment

# Get news for specific tickers
news_data = get_news_sentiment(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    limit=50
)
```

### Get News for Multiple Tickers

```python
from src.data.news_data import get_news_for_tickers

# Get news for multiple tickers over the last 30 days
tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
news_data = get_news_for_tickers(
    tickers=tickers,
    days_back=30,
    limit_per_ticker=10
)
```

### Get Market News

```python
from src.data.news_data import get_market_news

# Get market news for specific topics
topics = ['earnings', 'ipo', 'mergers']
market_news = get_market_news(
    topics=topics,
    days_back=7,
    limit=100
)
```

### Analyze Sentiment

```python
from src.data.news_data import analyze_news_sentiment

# Analyze sentiment distribution
analysis = analyze_news_sentiment(news_data)

print("Sentiment Distribution:")
for sentiment, count in analysis['sentiment_distribution'].items():
    percentage = analysis['sentiment_percentages'][sentiment]
    print(f"{sentiment}: {count} articles ({percentage}%)")
```

### Generate News Summary

```python
from src.data.news_data import get_news_summary

# Get summary of top articles
summary = get_news_summary(news_data, top_n=5)

for article in summary['top_articles']:
    print(f"Title: {article['title']}")
    print(f"Source: {article['source']}")
    print(f"Sentiment: {article['sentiment_label']}")
    print(f"Relevance Score: {article['relevance_score']}")
    print("---")
```

## API Reference

### `get_news_sentiment()`

Downloads news sentiment data from Alpha Vantage.

**Parameters:**
- `tickers` (list, optional): List of stock ticker symbols
- `topics` (list, optional): List of news topics
- `time_from` (str, optional): Start time (YYYYMMDDTHHMM format)
- `time_to` (str, optional): End time (YYYYMMDDTHHMM format)
- `limit` (int): Maximum number of articles (default: 50)

**Returns:**
- `pd.DataFrame`: News data with sentiment analysis

### `get_news_for_tickers()`

Gets news data for specific tickers over a time period.

**Parameters:**
- `tickers` (list): List of stock ticker symbols
- `days_back` (int): Number of days to look back (default: 30)
- `limit_per_ticker` (int): Articles per ticker (default: 10)

**Returns:**
- `pd.DataFrame`: Combined news data for all tickers

### `get_market_news()`

Gets general market news without specific ticker focus.

**Parameters:**
- `topics` (list, optional): List of news topics
- `days_back` (int): Number of days to look back (default: 7)
- `limit` (int): Maximum number of articles (default: 100)

**Returns:**
- `pd.DataFrame`: Market news data

### `analyze_news_sentiment()`

Analyzes sentiment distribution in news data.

**Parameters:**
- `news_data` (pd.DataFrame): News data to analyze

**Returns:**
- `dict`: Analysis results with sentiment distribution and statistics

### `get_news_summary()`

Creates a summary of the most relevant news articles.

**Parameters:**
- `news_data` (pd.DataFrame): News data to summarize
- `top_n` (int): Number of top articles to include (default: 5)

**Returns:**
- `dict`: Summary of top news articles

## Data Structure

The news data DataFrame contains the following columns:

- `ticker`: Stock ticker symbol
- `relevance_score`: Relevance score of the news (0-1)
- `ticker_sentiment_score`: Sentiment score for the ticker (-1 to 1)
- `ticker_sentiment_label`: Sentiment label (positive/negative/neutral)
- `time_published`: Publication time
- `title`: News title
- `url`: News URL
- `summary`: News summary
- `source`: News source

## Caching

The utility automatically caches downloaded data to reduce API calls:

- Cache files are stored in `news_cache/` directory
- Cache keys are based on search parameters (tickers, topics, time range, limit)
- Cache files use SHA256 hashes for unique identification
- Invalid or empty cache files are automatically re-downloaded

## Error Handling

The utility includes comprehensive error handling:

- API key validation
- Network error handling
- Rate limiting compliance
- Data validation and cleaning
- Graceful fallbacks for missing data

## Example Script

Run the example script to see the utility in action:

```bash
python examples/news_data_example.py
```

Make sure to set your `ALPHA_VANTAGE_API_KEY` environment variable first.

## Integration with Hyper Framework

The news data utility integrates seamlessly with the Hyper framework:

```python
from src.data import get_news_sentiment, analyze_news_sentiment

# Use in your ML pipeline
news_data = get_news_sentiment(tickers=['AAPL', 'MSFT'])
sentiment_features = analyze_news_sentiment(news_data)

# Combine with stock data for enhanced predictions
from src.data import get_stock_data
stock_data = get_stock_data(['AAPL', 'MSFT'])
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: Alpha Vantage API key not found.
   ```
   Solution: Set the `ALPHA_VANTAGE_API_KEY` environment variable.

2. **Rate Limit Exceeded**
   ```
   Error: API rate limit exceeded
   ```
   Solution: The utility includes automatic rate limiting. Wait for the next minute or upgrade to a premium API key.

3. **No Data Returned**
   ```
   Warning: No news data returned from Alpha Vantage API
   ```
   Solution: Check your search parameters and ensure the tickers/topics are valid.

4. **Cache Issues**
   ```
   Error loading from cache
   ```
   Solution: The utility will automatically re-download data if cache files are corrupted.

### Performance Tips

- Use caching to reduce API calls
- Batch ticker requests when possible
- Use appropriate time ranges to limit data size
- Consider upgrading to premium API for higher rate limits

## Contributing

When contributing to the news data utility:

1. Follow the existing code style
2. Add comprehensive error handling
3. Include rate limiting considerations
4. Update documentation for new features
5. Add tests for new functionality 