# Polygon.io News Data Integration

This document describes the new Polygon.io news data integration for the Hyper framework, providing an alternative to Alpha Vantage for news sentiment analysis.

## Overview

The Polygon.io integration provides:
- **Alternative news source**: Access to Polygon's comprehensive news database
- **Pagination support**: Efficient handling of large datasets
- **Sentiment analysis**: Calculated sentiment scores and labels
- **Unified interface**: Seamless switching between Alpha Vantage and Polygon
- **Caching**: Local caching for improved performance

## Setup

### 1. Install Dependencies

The Polygon client library is already included in `requirements.txt`:
```bash
pip install polygon-api-client>=1.12.0
```

### 2. Configure API Key

Create or update `src/config/polygon_key.py`:
```python
# Get your API key from https://polygon.io/
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY_HERE"
```

Or set environment variable:
```bash
export POLYGON_API_KEY="your_api_key_here"
```

### 3. Select News Provider

```python
from src.config.news_config import NewsProvider
from src.data.unified_news_data import set_news_provider

# Set Polygon as default provider
set_news_provider(NewsProvider.POLYGON)

# Or use Alpha Vantage
set_news_provider(NewsProvider.ALPHA_VANTAGE)
```

## Usage

### Basic News Retrieval

```python
from src.data.unified_news_data import get_news_for_tickers

# Get news for specific tickers
news_data = get_news_for_tickers(
    tickers=['AAPL', 'MSFT'],
    days_back=30,
    limit_per_ticker=10
)

print(f"Retrieved {len(news_data)} articles")
```

### Market News

```python
from src.data.unified_news_data import get_market_news

# Get general market news
market_news = get_market_news(
    days_back=7,
    limit=100
)
```

### Sentiment Analysis

```python
from src.data.unified_news_data import analyze_news_sentiment, get_news_summary

# Analyze sentiment distribution
sentiment_analysis = analyze_news_sentiment(news_data)
print("Sentiment distribution:", sentiment_analysis['sentiment_distribution'])

# Get news summary
summary = get_news_summary(news_data, top_n=5)
for article in summary['top_articles']:
    print(f"- {article['title']} ({article['sentiment_label']})")
```

### Provider Comparison

```python
from src.data.unified_news_data import compare_providers

# Compare both providers
comparison = compare_providers(
    tickers=['AAPL', 'MSFT'],
    days_back=7,
    limit=50
)

print(f"Alpha Vantage articles: {comparison['alpha_vantage']['articles_count']}")
print(f"Polygon articles: {comparison['polygon']['articles_count']}")
print(f"Recommended: {comparison['summary']['recommended_provider']}")
```

## Data Format

The Polygon implementation returns data in the same format as Alpha Vantage:

| Column | Description |
|--------|-------------|
| `ticker` | Stock ticker symbol |
| `relevance_score` | Calculated relevance (0-1) |
| `ticker_sentiment_score` | Calculated sentiment (-1 to 1) |
| `ticker_sentiment_label` | Sentiment label (positive/negative/neutral) |
| `time_published` | Publication timestamp |
| `title` | Article title |
| `url` | Article URL |
| `summary` | Article summary/description |
| `source` | News source/author |
| `keywords` | Article keywords (Polygon only) |
| `image_url` | Article image URL (Polygon only) |

## Features

### Sentiment Calculation

The Polygon implementation calculates sentiment using:
- **Keyword analysis**: Positive/negative keyword matching
- **Context scoring**: Relevance to specific tickers
- **Label assignment**: Automatic positive/negative/neutral classification

### Caching

- **Local cache**: News data cached in `news_cache/` directory
- **Hash-based filenames**: Unique cache files based on search parameters
- **Automatic invalidation**: Cache refreshed when parameters change

### Pagination

- **Automatic pagination**: Handles large datasets efficiently
- **Rate limiting**: Respects API rate limits
- **Progress tracking**: Shows download progress

## API Differences

| Feature | Alpha Vantage | Polygon |
|---------|---------------|---------|
| **Topics** | ✅ Supported | ❌ Not available |
| **Ticker-specific** | ✅ Supported | ✅ Supported |
| **Market news** | ✅ Supported | ✅ Supported |
| **Sentiment scores** | ✅ Provided | ✅ Calculated |
| **Rate limits** | 5 calls/minute | Varies by plan |
| **Data freshness** | Real-time | Real-time |

## Testing

Run the test script to verify functionality:

```bash
python test_polygon_news.py
```

This will:
- Test both providers
- Compare data formats
- Validate API keys
- Show usage examples

## Configuration

### Provider Selection

```python
# In your code
from src.config.news_config import NewsProvider
from src.data.unified_news_data import set_news_provider

# Set default provider
set_news_provider(NewsProvider.POLYGON)

# Or override per call
news_data = get_news_for_tickers(
    tickers=['AAPL'],
    provider=NewsProvider.POLYGON
)
```

### Environment Variables

```bash
# Alpha Vantage
export ALPHA_VANTAGE_API_KEY="your_key"

# Polygon
export POLYGON_API_KEY="your_key"
```

## Error Handling

The implementation includes comprehensive error handling:

- **API key validation**: Checks for valid API keys
- **Rate limiting**: Automatic delays between requests
- **Network errors**: Graceful handling of connection issues
- **Data validation**: Ensures data format consistency

## Performance

### Caching Benefits

- **First request**: Downloads from API
- **Subsequent requests**: Loads from cache
- **Cache invalidation**: Automatic when parameters change

### Rate Limiting

- **Alpha Vantage**: 12-second delays between calls
- **Polygon**: 1-second delays (adjustable based on plan)

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   ❌ Error: Polygon API key not found.
   ```
   - Check `src/config/polygon_key.py`
   - Verify environment variable

2. **No Data Returned**
   ```
   ⚠️  No news data returned from Polygon API
   ```
   - Check API key validity
   - Verify ticker symbols
   - Check date range

3. **Rate Limit Exceeded**
   ```
   ❌ API error: 429
   ```
   - Wait before retrying
   - Check API plan limits

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Migration from Alpha Vantage

To switch from Alpha Vantage to Polygon:

1. **Update configuration**:
   ```python
   set_news_provider(NewsProvider.POLYGON)
   ```

2. **Test functionality**:
   ```python
   # Compare providers
   comparison = compare_providers(['AAPL'], days_back=7)
   ```

3. **Update API keys**:
   - Get Polygon API key from https://polygon.io/
   - Update `src/config/polygon_key.py`

4. **Verify data format**:
   - Both providers return identical DataFrame format
   - No code changes needed for existing analysis

## Support

For issues with:
- **Polygon API**: Contact Polygon.io support
- **Implementation**: Check this documentation
- **Configuration**: Verify API keys and settings 