# Polygon News Downloader

A command-line tool for downloading Polygon news data for large cap tickers, similar to the Alpha Vantage OHLC downloader.

## Features

- **Full History Download**: Download all news data after a specified date
- **Incremental Updates**: Download only new articles since last update
- **Market News**: Download general market news (not ticker-specific)
- **Flexible Ticker Selection**: Use specific tickers or large cap tickers
- **Caching**: Intelligent caching with metadata tracking
- **Force Download**: Override cache and re-download data

## Usage

### Basic Usage

```bash
# Download incremental news for 5 large cap tickers (default)
python download_polygon_news.py --ticker-count 5

# Download full history for specific tickers
python download_polygon_news.py --tickers AAPL,MSFT,GOOGL --full

# Download market news for the last 7 days
python download_polygon_news.py --market-news --days-back 7
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tickers` | str | None | Comma-separated list of tickers (overrides --ticker-count) |
| `--ticker-count` | int | 1 | Number of tickers from get_large_cap_tickers (-1 for all) |
| `--full` | flag | False | Download full history (default: incremental) |
| `--start-date` | str | 2024-01-01 | Start date for news download (YYYY-MM-DD) |
| `--end-date` | str | None | End date for news download (defaults to now) |
| `--days-back` | int | 30 | Days to look back for incremental download |
| `--limit-per-ticker` | int | 1000 | Maximum articles per ticker |
| `--force` | flag | False | Force re-download even if cache exists |
| `--market-news` | flag | False | Download general market news |

## Examples

### 1. Incremental Download (Default)

Download new articles for the last 30 days for 3 large cap tickers:

```bash
python download_polygon_news.py --ticker-count 3
```

Output:
```
[INFO] Using tickers: ['AAPL', 'MSFT', 'GOOGL']
[INFO] Running incremental Polygon news download for: ['AAPL', 'MSFT', 'GOOGL']
[INFO] Looking back 30 days
[SUCCESS] Downloaded 45 new articles, total 1250 articles for 3/3 tickers
```

### 2. Full History Download

Download all news data since 2023 for specific tickers:

```bash
python download_polygon_news.py --tickers AAPL,MSFT,GOOGL --full --start-date 2023-01-01
```

Output:
```
[INFO] Using tickers: ['AAPL', 'MSFT', 'GOOGL']
[INFO] Running full Polygon news download for: ['AAPL', 'MSFT', 'GOOGL']
[INFO] Date range: 2023-01-01 to now
[SUCCESS] Downloaded 3500 total articles for 3/3 tickers
```

### 3. Market News Download

Download general market news for the last 7 days:

```bash
python download_polygon_news.py --market-news --days-back 7
```

Output:
```
[INFO] Using tickers: []
[INFO] Running Polygon market news download for the last 7 days
[SUCCESS] Downloaded 150 market news articles
```

### 4. Force Re-download

Force re-download even if cached data exists:

```bash
python download_polygon_news.py --tickers AAPL,MSFT --full --force
```

### 5. Custom Date Range

Download news for a specific date range:

```bash
python download_polygon_news.py --tickers AAPL,MSFT --full --start-date 2024-01-01 --end-date 2024-01-31
```

### 6. All Large Cap Tickers

Download for all available large cap tickers:

```bash
python download_polygon_news.py --ticker-count -1 --full
```

## Configuration

### Polygon API Key

Make sure your Polygon API key is configured in `src/config/polygon_key.py`:

```python
POLYGON_API_KEY = "your_api_key_here"
USE_CACHED_DATA = True
USE_MASTER_FILE = True
CACHE_DIR = "polygon_news_data"
```

### Cache Behavior

- **Incremental Mode**: Only downloads new articles since last update
- **Full Mode**: Downloads all articles in the specified date range
- **Cache Persistence**: Historical data remains valid indefinitely
- **Force Download**: Override cache and re-download all data

## Output Structure

### Downloaded Data

News data is stored in the following structure:

```
polygon_news_data/
├── master_metadata.json          # Master tracking file
├── AAPL/
│   ├── metadata.json            # Ticker-specific metadata
│   └── news_data.pkl           # Cached news data
├── MSFT/
│   ├── metadata.json
│   └── news_data.pkl
└── GOOGL/
    ├── metadata.json
    └── news_data.pkl
```

### Data Format

Each news article contains:

- `ticker`: Stock ticker symbol
- `title`: Article title
- `summary`: Article summary/description
- `time_published`: Publication timestamp
- `url`: Article URL
- `source`: News source
- `ticker_sentiment_score`: Sentiment score (-1 to 1)
- `ticker_sentiment_label`: Sentiment label (positive/negative/neutral)
- `relevance_score`: Relevance score (0 to 1)
- And more metadata fields

## Error Handling

The downloader includes comprehensive error handling:

- **API Rate Limits**: Automatic rate limiting and retry logic
- **Network Errors**: Graceful handling of connection issues
- **Missing Data**: Clear error messages for missing tickers
- **Cache Issues**: Automatic cache validation and recovery

## Performance Considerations

### API Usage

- **Rate Limiting**: Built-in delays to respect Polygon API limits
- **Batch Processing**: Downloads data in batches to minimize API calls
- **Caching**: Reduces API usage through intelligent caching

### Storage

- **Per-Ticker Storage**: Separate files for each ticker
- **Metadata Tracking**: Efficient tracking of downloaded data
- **Compression**: Data stored in efficient pickle format

## Troubleshooting

### Common Issues

1. **API Key Error**: Verify Polygon API key in configuration
2. **No Data Downloaded**: Check date range and ticker availability
3. **Rate Limit Errors**: Wait and retry, or reduce batch size
4. **Cache Issues**: Use `--force` to re-download data

### Debug Information

Enable verbose logging to see detailed progress:

```bash
python download_polygon_news.py --tickers AAPL --full --start-date 2024-01-01
```

## Integration

This downloader can be integrated into automated workflows:

```bash
# Daily incremental updates
python download_polygon_news.py --ticker-count 10

# Weekly full updates
python download_polygon_news.py --ticker-count 10 --full --start-date 2024-01-01

# Market news monitoring
python download_polygon_news.py --market-news --days-back 1
```

## Comparison with Alpha Vantage Downloader

| Feature | Alpha Vantage OHLC | Polygon News |
|---------|-------------------|--------------|
| Data Type | OHLC Price Data | News Articles |
| Incremental | ✅ | ✅ |
| Full History | ✅ | ✅ |
| Caching | ✅ | ✅ |
| Ticker Selection | ✅ | ✅ |
| Market Data | ❌ | ✅ |
| Sentiment Analysis | ❌ | ✅ | 