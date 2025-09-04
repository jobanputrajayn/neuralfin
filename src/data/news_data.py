"""
News data loading and processing utilities for the Hyper framework.

Contains functions for downloading news data from Alpha Vantage, caching, and news processing.
"""

import pandas as pd
import numpy as np
import hashlib
import os
from pathlib import Path
from etils import epath
from alpha_vantage.alphaintelligence import AlphaIntelligence
from typing import List, Dict, Optional, Union
import time
from datetime import datetime, timedelta
# --- Data Caching Configuration ---
NEWS_CACHE_DIR = epath.Path(Path.cwd() / 'news_cache')

# Alpha Vantage API configuration - try config file first, then environment variable
try:
    from src.config.av_key import ALPHA_VANTAGE_API_KEY
except ImportError:
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

if not ALPHA_VANTAGE_API_KEY:
    print("⚠️  Warning: ALPHA_VANTAGE_API_KEY not found in config or environment variable.")
    print("   Please set your Alpha Vantage API key in src/config/av_key.py or as environment variable.")


def _get_news_cache_filename(tickers, topics, time_from, time_to, limit):
    """
    Generates a unique cache filename based on news search parameters.
    
    Args:
        tickers: List of ticker symbols
        topics: List of news topics
        time_from: Start time for news search
        time_to: End time for news search
        limit: Maximum number of news articles
        
    Returns:
        Path: Cache file path
    """
    # Create a consistent string representation of the parameters
    ticker_string = "_".join(sorted(tickers)) if tickers else "all"
    topic_string = "_".join(sorted(topics)) if topics else "all"
    time_string = f"{time_from}_{time_to}" if time_from and time_to else "all"
    
    # Combine all parameters into a single string
    unique_string = f"{ticker_string}_{topic_string}_{time_string}_limit{limit}"
    # Use SHA256 hash to create a short, unique filename
    hash_object = hashlib.sha256(unique_string.encode())
    return NEWS_CACHE_DIR / f"{hash_object.hexdigest()}.pkl"


def _validate_alpha_vantage_key():
    """
    Validates that Alpha Vantage API key is available.
    
    Returns:
        bool: True if API key is available, False otherwise
    """
    if not ALPHA_VANTAGE_API_KEY:
        print("❌ Error: Alpha Vantage API key not found.")
        print("   Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        return False
    return True


def _rate_limit_delay():
    """
    Implements rate limiting for Alpha Vantage API calls.
    Alpha Vantage has a limit of 5 API calls per minute for free tier.
    """
    time.sleep(12)  # Wait 12 seconds between calls to stay under limit


def get_news_sentiment(tickers: Optional[List[str]] = None, 
                      topics: Optional[List[str]] = None,
                      time_from: Optional[str] = None,
                      time_to: Optional[str] = None,
                      limit: int = 50) -> pd.DataFrame:
    """
    Downloads news sentiment data from Alpha Vantage for specified tickers and topics.
    
    Args:
        tickers (list, optional): List of stock ticker symbols to search for.
        topics (list, optional): List of news topics to search for.
        time_from (str, optional): Start time for news search (YYYY-MM-DD format).
        time_to (str, optional): End time for news search (YYYY-MM-DD format).
        limit (int): Maximum number of news articles to retrieve (default: 50).
        
    Returns:
        pd.DataFrame: DataFrame containing news sentiment data with columns:
                     - ticker: Stock ticker symbol
                     - relevance_score: Relevance score of the news
                     - ticker_sentiment_score: Sentiment score for the ticker
                     - ticker_sentiment_label: Sentiment label (positive/negative/neutral)
                     - time_published: Publication time
                     - title: News title
                     - url: News URL
                     - summary: News summary
                     - source: News source
    """
    if not _validate_alpha_vantage_key():
        return pd.DataFrame()
    
    cache_file = _get_news_cache_filename(tickers, topics, time_from, time_to, limit)
    
    if cache_file.exists():
        print(f"Loading news data from cache: {cache_file}")
        try:
            cached_data = pd.read_pickle(cache_file)
            if not cached_data.empty:
                return cached_data
            else:
                print(f"  Warning: Cached file {cache_file} is empty. Re-downloading.")
        except Exception as e:
            print(f"  Error loading from cache {cache_file}: {e}. Re-downloading.")
    
    print(f"Downloading news sentiment data...")
    print(f"  Tickers: {tickers if tickers else 'All'}")
    print(f"  Topics: {topics if topics else 'All'}")
    print(f"  Time range: {time_from} to {time_to if time_to else 'Now'}")
    print(f"  Limit: {limit}")
    
    try:
        alpha_intelligence = AlphaIntelligence(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        
        # Prepare tickers string for API call
        tickers_str = ','.join(tickers) if tickers else None
        
        # Prepare topics string for API call
        topics_str = ','.join(topics) if topics else None
        
        # Make API call
        try:
            data, meta_data, _ = alpha_intelligence.get_news_sentiment(
                tickers=tickers_str,
                topics=topics_str,
                time_from=time_from,
                time_to=time_to,
                limit=limit
            )
        except Exception as api_error:
            print(f"❌ API call error: {api_error}")
            return pd.DataFrame()
        
        # Rate limiting
        _rate_limit_delay()
        
        # Check if data is valid
        if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
            # Clean and process the data
            processed_data = _process_news_data(data)
            
            # Save to cache
            NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            processed_data.to_pickle(cache_file)
            print(f"News data saved to cache: {cache_file}")
            
            print(f"✅ Successfully downloaded news data: {len(processed_data)} articles")
            return processed_data
        else:
            print("⚠️  No news data returned from Alpha Vantage API")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error downloading news data: {e}")
        return pd.DataFrame()


def _process_news_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and cleans news data from Alpha Vantage.
    
    Args:
        data: Raw news data from Alpha Vantage
        
    Returns:
        pd.DataFrame: Processed and cleaned news data
    """
    try:
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Check if this is the Alpha Vantage format with nested sentiment data
        if 'ticker_sentiment' in data.columns:
            print("📊 Processing Alpha Vantage nested sentiment format...")
            return _process_nested_sentiment_data(data)
        
        # Original flat format processing
        required_columns = [
            'ticker', 'relevance_score', 'ticker_sentiment_score', 
            'ticker_sentiment_label', 'time_published', 'title', 
            'url', 'summary', 'source'
        ]
        
        # Check which columns are missing
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"⚠️  Warning: Missing columns in news data: {missing_columns}")
            return pd.DataFrame()
        
        # Convert numeric columns
        numeric_columns = ['relevance_score', 'ticker_sentiment_score']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Convert time_published to datetime
        if 'time_published' in data.columns:
            data['time_published'] = pd.to_datetime(data['time_published'], errors='coerce')
        
        # Remove rows with missing critical data
        critical_columns = ['ticker', 'title', 'time_published']
        data = data.dropna(subset=critical_columns)
        
        # Sort by time_published (most recent first)
        if 'time_published' in data.columns:
            data = data.sort_values('time_published', ascending=False)
        
        # Reset index
        data = data.reset_index(drop=True)
        
        return data
        
    except Exception as e:
        print(f"❌ Error processing news data: {e}")
        return pd.DataFrame()


def _process_nested_sentiment_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes Alpha Vantage data with nested sentiment information.
    
    Args:
        data: Raw news data with nested sentiment
        
    Returns:
        pd.DataFrame: Flattened and processed news data
    """
    try:
        processed_rows = []
        
        for idx, row in data.iterrows():
            # Extract basic article information
            article_data = {
                'title': row.get('title', ''),
                'url': row.get('url', ''),
                'summary': row.get('summary', ''),
                'source': row.get('source', ''),
                'time_published': row.get('time_published', ''),
                'topics': row.get('topics', ''),
                'authors': row.get('authors', ''),
                'banner_image': row.get('banner_image', ''),
                'overall_sentiment_score': row.get('overall_sentiment_score', 0),
                'overall_sentiment_label': row.get('overall_sentiment_label', 'neutral')
            }
            
            # Process nested ticker sentiment data
            ticker_sentiment = row.get('ticker_sentiment', [])
            
            if isinstance(ticker_sentiment, list) and ticker_sentiment:
                # Multiple tickers for this article
                for ticker_data in ticker_sentiment:
                    if isinstance(ticker_data, dict):
                        row_data = article_data.copy()
                        row_data.update({
                            'ticker': ticker_data.get('ticker', ''),
                            'relevance_score': ticker_data.get('relevance_score', 0),
                            'ticker_sentiment_score': ticker_data.get('ticker_sentiment_score', 0),
                            'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label', 'neutral')
                        })
                        processed_rows.append(row_data)
            else:
                # No ticker sentiment data, create a single row
                row_data = article_data.copy()
                row_data.update({
                    'ticker': '',
                    'relevance_score': 0,
                    'ticker_sentiment_score': 0,
                    'ticker_sentiment_label': 'neutral'
                })
                processed_rows.append(row_data)
        
        if not processed_rows:
            print("⚠️  No valid sentiment data found")
            return pd.DataFrame()
        
        # Create DataFrame from processed rows
        processed_data = pd.DataFrame(processed_rows)
        
        # Convert numeric columns
        numeric_columns = ['relevance_score', 'ticker_sentiment_score', 'overall_sentiment_score']
        for col in numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Convert time_published to datetime
        if 'time_published' in processed_data.columns:
            processed_data['time_published'] = pd.to_datetime(processed_data['time_published'], errors='coerce')
        
        # Remove rows with missing critical data
        critical_columns = ['title', 'time_published']
        processed_data = processed_data.dropna(subset=critical_columns)
        
        # Sort by time_published (most recent first)
        if 'time_published' in processed_data.columns:
            processed_data = processed_data.sort_values('time_published', ascending=False)
        
        # Reset index
        processed_data = processed_data.reset_index(drop=True)
        
        print(f"✅ Processed {len(processed_data)} sentiment entries from {len(data)} articles")
        return processed_data
        
    except Exception as e:
        print(f"❌ Error processing nested sentiment data: {e}")
        return pd.DataFrame()


def get_news_for_tickers(tickers: List[str], 
                        days_back: Optional[int] = None,
                        time_from: Optional[str] = None,
                        time_to: Optional[str] = None,
                        limit_per_ticker: int = 10) -> pd.DataFrame:
    """
    Gets news data for specific tickers over a specified time period.
    
    Args:
        tickers: List of stock ticker symbols
        days_back: Number of days to look back for news (default: 30 if no time_from/time_to provided)
        time_from: Start time for news search (YYYYMMDDTHHMM format)
        time_to: End time for news search (YYYYMMDDTHHMM format)
        limit_per_ticker: Maximum number of articles per ticker (default: 10)
        
    Returns:
        pd.DataFrame: Combined news data for all tickers
    """
    if not _validate_alpha_vantage_key():
        return pd.DataFrame()
    
    # Calculate time range if not provided
    if time_from is None or time_to is None:
        if days_back is None:
            days_back = 30
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        time_from = start_date.strftime('%Y%m%dT0000')
        time_to = end_date.strftime('%Y%m%dT2359')
    
    print(f"Downloading news for {len(tickers)} tickers from {time_from} to {time_to}...")
    
    all_news = []
    
    for ticker in tickers:
        print(f"  Downloading news for {ticker}...")
        try:
            ticker_news = get_news_sentiment(
                tickers=[ticker],
                time_from=time_from,
                time_to=time_to,
                limit=limit_per_ticker
            )
            
            if not ticker_news.empty:
                all_news.append(ticker_news)
                print(f"    ✅ Found {len(ticker_news)} articles for {ticker}")
            else:
                print(f"    ⚠️  No news found for {ticker}")
                
        except Exception as e:
            print(f"    ❌ Error downloading news for {ticker}: {e}")
    
    if all_news:
        combined_news = pd.concat(all_news, ignore_index=True)
        print(f"✅ Successfully downloaded {len(combined_news)} total articles")
        return combined_news
    else:
        print("❌ No news data collected for any ticker")
        return pd.DataFrame()


def get_market_news(topics: Optional[List[str]] = None,
                   days_back: int = 7,
                   limit: int = 100) -> pd.DataFrame:
    """
    Gets general market news without specific ticker focus.
    
    Args:
        topics: List of news topics (e.g., ['earnings', 'ipo', 'mergers'])
        days_back: Number of days to look back for news (default: 7)
        limit: Maximum number of articles to retrieve (default: 100)
        
    Returns:
        pd.DataFrame: Market news data
    """
    if not _validate_alpha_vantage_key():
        return pd.DataFrame()
    
    print(f"Downloading market news for the last {days_back} days...")
    
    # Calculate time range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    time_from = start_date.strftime('%Y%m%dT0000')
    time_to = end_date.strftime('%Y%m%dT2359')
    
    try:
        market_news = get_news_sentiment(
            topics=topics,
            time_from=time_from,
            time_to=time_to,
            limit=limit
        )
        
        if not market_news.empty:
            print(f"✅ Successfully downloaded {len(market_news)} market news articles")
        else:
            print("⚠️  No market news found")
            
        return market_news
        
    except Exception as e:
        print(f"❌ Error downloading market news: {e}")
        return pd.DataFrame()


def analyze_news_sentiment(news_data: pd.DataFrame) -> Dict:
    """
    Analyzes sentiment distribution in news data.
    
    Args:
        news_data: DataFrame containing news sentiment data
        
    Returns:
        Dict: Analysis results including sentiment distribution and statistics
    """
    if news_data.empty:
        return {}
    
    analysis = {}
    
    # Sentiment distribution
    if 'ticker_sentiment_label' in news_data.columns:
        sentiment_counts = news_data['ticker_sentiment_label'].value_counts()
        analysis['sentiment_distribution'] = sentiment_counts.to_dict()
        
        # Calculate sentiment percentages
        total_articles = len(news_data)
        sentiment_percentages = (sentiment_counts / total_articles * 100).round(2)
        analysis['sentiment_percentages'] = sentiment_percentages.to_dict()
    
    # Relevance score statistics
    if 'relevance_score' in news_data.columns:
        relevance_stats = news_data['relevance_score'].describe()
        analysis['relevance_statistics'] = relevance_stats.to_dict()
    
    # Ticker sentiment score statistics
    if 'ticker_sentiment_score' in news_data.columns:
        sentiment_stats = news_data['ticker_sentiment_score'].describe()
        analysis['sentiment_score_statistics'] = sentiment_stats.to_dict()
    
    # Time-based analysis
    if 'time_published' in news_data.columns:
        news_data['date'] = pd.to_datetime(news_data['time_published']).dt.date
        daily_counts = news_data['date'].value_counts().sort_index()
        analysis['daily_article_counts'] = daily_counts.to_dict()
    
    # Source analysis
    if 'source' in news_data.columns:
        source_counts = news_data['source'].value_counts().head(10)
        analysis['top_sources'] = source_counts.to_dict()
    
    return analysis


def get_news_summary(news_data: pd.DataFrame, top_n: int = 5) -> Dict:
    """
    Creates a summary of the most relevant news articles.
    
    Args:
        news_data: DataFrame containing news data
        top_n: Number of top articles to include in summary
        
    Returns:
        Dict: Summary of top news articles
    """
    if news_data.empty:
        return {}
    
    summary = {}
    
    # Sort by relevance score and get top articles
    if 'relevance_score' in news_data.columns:
        top_articles = news_data.nlargest(top_n, 'relevance_score')
        
        summary['top_articles'] = []
        for _, article in top_articles.iterrows():
            article_summary = {
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'time_published': str(article.get('time_published', '')),
                'relevance_score': article.get('relevance_score', 0),
                'sentiment_label': article.get('ticker_sentiment_label', ''),
                'sentiment_score': article.get('ticker_sentiment_score', 0),
                'url': article.get('url', ''),
                'summary': article.get('summary', '')[:200] + '...' if len(article.get('summary', '')) > 200 else article.get('summary', '')
            }
            summary['top_articles'].append(article_summary)
    
    # Overall statistics
    summary['total_articles'] = len(news_data)
    summary['date_range'] = {
        'earliest': str(news_data['time_published'].min()) if 'time_published' in news_data.columns else '',
        'latest': str(news_data['time_published'].max()) if 'time_published' in news_data.columns else ''
    }
    
    return summary 