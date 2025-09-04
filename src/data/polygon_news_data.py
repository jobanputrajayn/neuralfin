"""
Polygon.io news data loading and processing utilities for the Hyper framework.

Contains functions for downloading news data from Polygon.io, caching, and news processing.
"""

import pandas as pd
import numpy as np
import hashlib
import os
import json
from pathlib import Path
from etils import epath
from polygon import RESTClient
from typing import List, Dict, Optional, Union
import time
from datetime import datetime, timedelta
import requests

# --- Data Caching Configuration ---
NEWS_CACHE_DIR = epath.Path(Path.cwd() / 'news_cache')

# Polygon API configuration
try:
    from src.config.polygon_key import POLYGON_API_KEY, USE_CACHED_DATA, USE_MASTER_FILE, CACHE_DIR
except ImportError:
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    USE_CACHED_DATA = True
    USE_MASTER_FILE = True
    CACHE_DIR = "polygon_news_data"

# Enhanced cache directory structure
POLYGON_CACHE_DIR = epath.Path(Path.cwd() / CACHE_DIR)
MASTER_FILE_PATH = POLYGON_CACHE_DIR / "master_metadata.json"

if not POLYGON_API_KEY or POLYGON_API_KEY == "YOUR_POLYGON_API_KEY_HERE":
    print("⚠️  Warning: POLYGON_API_KEY not found in config or environment variable.")
    print("   Please set your Polygon API key in src/config/polygon_key.py or as environment variable.")


def _get_polygon_news_cache_filename(tickers, published_utc_gte, published_utc_lte, limit):
    """
    Generates a unique cache filename based on Polygon news search parameters.
    
    Args:
        tickers: List of ticker symbols
        published_utc_gte: Start time for news search
        published_utc_lte: End time for news search
        limit: Maximum number of news articles
        
    Returns:
        Path: Cache file path
    """
    # Create a consistent string representation of the parameters
    ticker_string = "_".join(sorted(tickers)) if tickers else "all"
    time_string = f"{published_utc_gte}_{published_utc_lte}" if published_utc_gte and published_utc_lte else "all"
    
    # Combine all parameters into a single string
    unique_string = f"polygon_{ticker_string}_{time_string}_limit{limit}"
    # Use SHA256 hash to create a short, unique filename
    hash_object = hashlib.sha256(unique_string.encode())
    return NEWS_CACHE_DIR / f"{hash_object.hexdigest()}.pkl"


def _validate_polygon_key():
    """
    Validates that Polygon API key is available.
    
    Returns:
        bool: True if API key is available, False otherwise
    """
    if not POLYGON_API_KEY or POLYGON_API_KEY == "YOUR_POLYGON_API_KEY_HERE":
        print("❌ Error: Polygon API key not found.")
        print("   Please set the POLYGON_API_KEY environment variable or update src/config/polygon_key.py")
        return False
    return True


def _rate_limit_delay():
    """
    Implements rate limiting for Polygon API calls.
    Polygon has different rate limits based on subscription tier.
    """
    time.sleep(12)  # More conservative delay for free tier


class PolygonClinet(RESTClient):
    from polygon.rest.models.request import RequestOptionBuilder
    def _paginate_iter(
            self,
            path: str,
            params: dict,
            deserializer,
            result_key: str = "results",
            options: Optional[RequestOptionBuilder] = None,
        ):
            from urllib.parse import urlparse
            while True:
                resp = self._get(
                    path=path,
                    params=params,
                    deserializer=deserializer,
                    result_key=result_key,
                    raw=True,
                    options=options,
                )

                try:
                    decoded = self._decode(resp)
                except ValueError as e:
                    print(f"Error decoding json response: {e}")
                    return []

                if result_key not in decoded:
                    return []
                for t in decoded[result_key]:
                    yield deserializer(t)
                if not self.pagination or "next_url" not in decoded:
                    return
                next_url = decoded["next_url"]
                parsed = urlparse(next_url)
                path = parsed.path
                if parsed.query:
                    path += "?" + parsed.query
                params = {}
                _rate_limit_delay()

def _fetch_polygon_news(tickers: Optional[List[str]] = None,
                       published_utc_gte: Optional[str] = None,
                       published_utc_lte: Optional[str] = None,
                       limit: int = 50) -> List:
    """
    Fetches news data from Polygon with custom pagination and rate limiting.
    Continues fetching until either no more data is available or published_utc_lte is reached.
    
    Args:
        tickers: List of ticker symbols
        published_utc_gte: Start time (ISO format)
        published_utc_lte: End time (ISO format)
        limit: Maximum number of articles per request (default: 50)
        
    Returns:
        List: List of news articles
    """
    if not _validate_polygon_key():
        return []
    
    try:
        client = PolygonClinet(POLYGON_API_KEY)
        
        all_articles = []
        current_limit = limit
        reached_end_date = False
        
        # Parse the end date for comparison if provided
        end_date = None
        if published_utc_lte:
            try:
                end_date = pd.to_datetime(published_utc_lte)
            except Exception:
                pass
        
        # Track the latest date we've seen for pagination
        current_gte = published_utc_gte
        
        print(f"Fetching Polygon news data with pagination...")
        print(f"  Tickers: {tickers if tickers else 'All'}")
        print(f"  Time range: {published_utc_gte} to {published_utc_lte if published_utc_lte else 'Now'}")
        print(f"  Limit per request: {limit}")
        
        while not reached_end_date:
            # Fetch a batch of articles with updated start date
            articles_iter = client.list_ticker_news(
                ticker=",".join(tickers) if tickers else None,
                published_utc_gte=current_gte,
                published_utc_lte=published_utc_lte,
                limit=current_limit
            )
            
            # Convert iterator to list
            batch_articles = list(articles_iter)
            
            if not batch_articles:
                print(f"  No more articles available")
                break
            
            print(f"  Fetched batch of {len(batch_articles)} articles")
            
            # Check if we've gone past the end date
            if end_date and batch_articles:
                # Check if any article in this batch is after the end date
                # (meaning we've gone past our time boundary)
                for article in batch_articles:
                    published_utc = getattr(article, 'published_utc', '')
                    if published_utc:
                        try:
                            article_date = pd.to_datetime(published_utc)
                            if article_date > end_date:
                                print(f"  Found article after end date {published_utc_lte}, stopping pagination")
                                reached_end_date = True
                                break
                        except Exception:
                            continue
            
            # Add articles to our collection
            all_articles.extend(batch_articles)
            
            # Update the start date for next iteration based on the latest article in this batch
            if batch_articles and not reached_end_date:
                latest_date = None
                for article in batch_articles:
                    published_utc = getattr(article, 'published_utc', '')
                    if published_utc:
                        try:
                            article_date = pd.to_datetime(published_utc)
                            if latest_date is None or article_date > latest_date:
                                latest_date = article_date
                        except Exception:
                            continue
                
                if latest_date:
                    # Set the next start date to be after the latest article we've seen
                    # Add 1 second to avoid duplicates
                    next_start = latest_date + pd.Timedelta(seconds=1)
                    current_gte = next_start.strftime('%Y-%m-%dT%H:%M:%SZ')
                    print(f"  Updated start date for next batch: {current_gte}")
            
            # If we got fewer articles than requested, we've reached the end
            if len(batch_articles) < current_limit:
                print(f"  Received fewer articles than requested ({len(batch_articles)} < {current_limit})")
                break
            
            # Rate limiting between requests
            _rate_limit_delay()
        
        print(f"✅ Total articles fetched: {len(all_articles)}")
        return all_articles
        
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "too many" in error_str:
            print(f"⚠️  Rate limit exceeded. Please wait before retrying.")
            raise e
        else:
            print(f"❌ Error fetching news data: {e}")
        return []


def get_polygon_news_sentiment(tickers: Optional[List[str]] = None,
                              published_utc_gte: Optional[str] = None,
                              published_utc_lte: Optional[str] = None,
                              limit: int = 50) -> pd.DataFrame:
    """
    Downloads news data from Polygon.io for specified tickers and time range.
    
    Args:
        tickers (list, optional): List of stock ticker symbols to search for.
        published_utc_gte (str, optional): Start time for news search (ISO format).
        published_utc_lte (str, optional): End time for news search (ISO format).
        limit (int): Maximum number of news articles to retrieve (default: 50).
        
    Returns:
        pd.DataFrame: DataFrame containing news data with columns matching Alpha Vantage format:
                     - ticker: Stock ticker symbol
                     - relevance_score: Relevance score (calculated from keywords)
                     - ticker_sentiment_score: Sentiment score (calculated from keywords)
                     - ticker_sentiment_label: Sentiment label (positive/negative/neutral)
                     - time_published: Publication time
                     - title: News title
                     - url: News URL
                     - summary: News summary
                     - source: News source
    """
    if not _validate_polygon_key():
        return pd.DataFrame()
    
    # Make time parameters time-agnostic
    if published_utc_gte:
        try:
            dt_gte = pd.to_datetime(published_utc_gte)
            published_utc_gte = dt_gte.strftime('%Y-%m-%d') + 'T00:00:00Z'
        except Exception:
            pass  # Keep original if parsing fails
    
    if published_utc_lte:
        try:
            dt_lte = pd.to_datetime(published_utc_lte)
            published_utc_lte = dt_lte.strftime('%Y-%m-%d') + 'T23:59:59Z'
        except Exception:
            pass  # Keep original if parsing fails
    
    cache_file = _get_polygon_news_cache_filename(tickers, published_utc_gte, published_utc_lte, limit)
    
    if cache_file.exists():
        print(f"Loading Polygon news data from cache: {cache_file}")
        try:
            cached_data = pd.read_pickle(cache_file)
            if not cached_data.empty:
                return cached_data
            else:
                print(f"  Warning: Cached file {cache_file} is empty. Re-downloading.")
        except Exception as e:
            print(f"  Error loading from cache {cache_file}: {e}. Re-downloading.")
    
    print(f"Downloading Polygon news data...")
    print(f"  Tickers: {tickers if tickers else 'All'}")
    print(f"  Time range: {published_utc_gte} to {published_utc_lte if published_utc_lte else 'Now'}")
    print(f"  Limit: {limit}")
    
    try:
        # Fetch news data using built-in pagination
        raw_articles = _fetch_polygon_news(
            tickers=tickers,
            published_utc_gte=published_utc_gte,
            published_utc_lte=published_utc_lte,
            limit=limit
        )
        
        if raw_articles:
            # Process and convert to DataFrame
            processed_data = _process_polygon_news_data(raw_articles, tickers)
            
            # Save to cache
            NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            processed_data.to_pickle(cache_file)
            print(f"Polygon news data saved to cache: {cache_file}")
            
            print(f"✅ Successfully downloaded Polygon news data: {len(processed_data)} articles")
            return processed_data
        else:
            print("⚠️  No news data returned from Polygon API")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error downloading Polygon news data: {e}")
        return pd.DataFrame()


def _process_polygon_news_data(articles: List, target_tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Processes and converts Polygon news data to match Alpha Vantage format.
    
    Args:
        articles: List of TickerNews objects from Polygon API
        target_tickers: List of target tickers for relevance calculation
        
    Returns:
        pd.DataFrame: Processed news data in Alpha Vantage format
    """
    try:
        processed_rows = []
        
        for article in articles:
            # Extract basic article information from TickerNews object
            title = getattr(article, 'title', '')
            description = getattr(article, 'description', '')
            author = getattr(article, 'author', '')
            published_utc = getattr(article, 'published_utc', '')
            article_url = getattr(article, 'article_url', '')
            image_url = getattr(article, 'image_url', '')
            keywords = getattr(article, 'keywords', [])
            
            # Ensure keywords is a list
            if keywords is None:
                keywords = []
            elif not isinstance(keywords, list):
                keywords = [str(keywords)] if keywords else []
            
            # Extract tickers mentioned in the article
            tickers_mentioned = getattr(article, 'tickers', [])
            
            # Ensure tickers_mentioned is a list
            if tickers_mentioned is None:
                tickers_mentioned = []
            elif not isinstance(tickers_mentioned, list):
                tickers_mentioned = [str(tickers_mentioned)] if tickers_mentioned else []
            
            # Calculate relevance and sentiment for each ticker
            if target_tickers:
                # If specific tickers requested, only process those
                relevant_tickers = [t for t in tickers_mentioned if t in target_tickers]
            else:
                # Process all mentioned tickers
                relevant_tickers = tickers_mentioned
            
            # Check for sentiment insights
            insights = getattr(article, 'insights', [])
            
            # Ensure insights is a list
            if insights is None:
                insights = []
            elif not isinstance(insights, list):
                insights = [insights] if insights else []
            has_sentiment_insight = False
            sentiment_score = 0.0
            sentiment_label = 'neutral'
            sentiment_reasoning = ''
            
            # Use sentiment from insights if available
            try:
                if insights and isinstance(insights, list):
                    # Find insight for the current ticker or use first available
                    ticker_insight = None
                    for insight in insights:
                        if hasattr(insight, 'ticker') and getattr(insight, 'ticker', '') in relevant_tickers:
                            ticker_insight = insight
                            break
                    
                    # If no specific ticker insight found, use first available
                    if not ticker_insight and insights:
                        ticker_insight = insights[0]
                    
                    if ticker_insight:
                        sentiment_label = getattr(ticker_insight, 'sentiment', 'neutral')
                        sentiment_reasoning = getattr(ticker_insight, 'sentiment_reasoning', '')
                        has_sentiment_insight = True
                        
                        # Convert sentiment label to score
                        if sentiment_label == 'positive':
                            sentiment_score = 0.5
                        elif sentiment_label == 'negative':
                            sentiment_score = -0.5
                        else:
                            sentiment_score = 0.0
                        
            except Exception:
                # Fall back to calculated sentiment
                pass
            
            if not relevant_tickers:
                # If no specific tickers mentioned, create a general entry
                relevance_score = _calculate_relevance_score(title, description, keywords, [])
                
                # Use calculated sentiment if no insight available
                if not has_sentiment_insight:
                    sentiment_score = _calculate_sentiment_score(title, description, keywords)
                    sentiment_label = _get_sentiment_label(sentiment_score)
                
                # Get sentiment reasoning if available
                sentiment_reasoning = ''
                if has_sentiment_insight and hasattr(insights, 'sentiment_reasoning'):
                    sentiment_reasoning = getattr(insights, 'sentiment_reasoning', '')
                
                row_data = {
                    'ticker': '',
                    'relevance_score': relevance_score,
                    'ticker_sentiment_score': sentiment_score,
                    'ticker_sentiment_label': sentiment_label,
                    'overall_sentiment_score': sentiment_score,  # Use ticker sentiment as overall
                    'overall_sentiment_label': sentiment_label,  # Use ticker sentiment as overall
                    'time_published': published_utc,
                    'title': title,
                    'url': article_url,
                    'summary': description,
                    'source': author,
                    'topics': ', '.join(keywords) if keywords else '',
                    'authors': author,
                    'banner_image': image_url,
                    'keywords': keywords,
                    'image_url': image_url,
                    'sentiment_reasoning': sentiment_reasoning,
                    'date': pd.to_datetime(published_utc).date() if published_utc else None
                }
                processed_rows.append(row_data)
            else:
                # Process each relevant ticker
                for ticker in relevant_tickers:
                    relevance_score = _calculate_relevance_score(title, description, keywords, [ticker])
                    
                    # Find specific insight for this ticker
                    ticker_sentiment_score = sentiment_score
                    ticker_sentiment_label = sentiment_label
                    ticker_sentiment_reasoning = sentiment_reasoning
                    
                    if has_sentiment_insight and insights:
                        for insight in insights:
                            if hasattr(insight, 'ticker') and getattr(insight, 'ticker', '') == ticker:
                                ticker_sentiment_label = getattr(insight, 'sentiment', 'neutral')
                                ticker_sentiment_reasoning = getattr(insight, 'sentiment_reasoning', '')
                                
                                # Convert sentiment label to score
                                if ticker_sentiment_label == 'positive':
                                    ticker_sentiment_score = 0.5
                                elif ticker_sentiment_label == 'negative':
                                    ticker_sentiment_score = -0.5
                                else:
                                    ticker_sentiment_score = 0.0
                                break
                    else:
                        # Use calculated sentiment if no insight available
                        ticker_sentiment_score = _calculate_sentiment_score(title, description, keywords)
                        ticker_sentiment_label = _get_sentiment_label(ticker_sentiment_score)
                    
                    row_data = {
                        'ticker': ticker,
                        'relevance_score': relevance_score,
                        'ticker_sentiment_score': ticker_sentiment_score,
                        'ticker_sentiment_label': ticker_sentiment_label,
                        'overall_sentiment_score': ticker_sentiment_score,  # Use ticker sentiment as overall
                        'overall_sentiment_label': ticker_sentiment_label,  # Use ticker sentiment as overall
                        'time_published': published_utc,
                        'title': title,
                        'url': article_url,
                        'summary': description,
                        'source': author,
                        'topics': ', '.join(keywords) if keywords else '',
                        'authors': author,
                        'banner_image': image_url,
                        'keywords': keywords,
                        'image_url': image_url,
                        'sentiment_reasoning': ticker_sentiment_reasoning,
                        'date': pd.to_datetime(published_utc).date() if published_utc else None
                    }
                    processed_rows.append(row_data)
        
        if not processed_rows:
            print("⚠️  No valid news data found")
            return pd.DataFrame()
        
        # Create DataFrame from processed rows
        processed_data = pd.DataFrame(processed_rows)
        
        # Convert numeric columns
        numeric_columns = ['relevance_score', 'ticker_sentiment_score', 'overall_sentiment_score']
        for col in numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Convert time_published to datetime and extract just the date
        if 'time_published' in processed_data.columns:
            processed_data['time_published'] = pd.to_datetime(processed_data['time_published'], errors='coerce')
            # Extract just the date part (time-agnostic)
            processed_data['date'] = processed_data['time_published'].dt.date
        
        # Remove rows with missing critical data
        critical_columns = ['title', 'time_published']
        processed_data = processed_data.dropna(subset=critical_columns)
        
        # Sort by time_published (most recent first)
        if 'time_published' in processed_data.columns:
            processed_data = processed_data.sort_values('time_published', ascending=False)
        
        # Reset index
        processed_data = processed_data.reset_index(drop=True)
        
        print(f"✅ Processed {len(processed_data)} news entries from {len(articles)} articles")
        return processed_data
        
    except Exception as e:
        print(f"❌ Error processing Polygon news data: {e}")
        import traceback
        print(f"❌ Full error details: {traceback.format_exc()}")
        return pd.DataFrame()


def _calculate_relevance_score(title: str, description: str, keywords: List[str], target_tickers: List[str]) -> float:
    """
    Calculate relevance score based on ticker mentions and keyword matches.
    
    Args:
        title: Article title
        description: Article description
        keywords: Article keywords
        target_tickers: List of target ticker symbols
        
    Returns:
        float: Relevance score between 0 and 1
    """
    if not target_tickers:
        return 0.5  # Default relevance for general articles
    
    text = f"{title} {description}".lower()
    keyword_text = " ".join(keywords).lower()
    combined_text = f"{text} {keyword_text}"
    
    # Count ticker mentions
    ticker_mentions = sum(1 for ticker in target_tickers if ticker.lower() in combined_text)
    
    # Calculate base relevance
    base_relevance = min(ticker_mentions * 0.3, 1.0)
    
    # Add keyword relevance
    keyword_relevance = min(len([k for k in keywords if k.lower() in combined_text]) * 0.1, 0.5)
    
    return min(base_relevance + keyword_relevance, 1.0)


def _calculate_sentiment_score(title: str, description: str, keywords: List[str]) -> float:
    """
    Calculate sentiment score based on keywords and text analysis.
    
    Args:
        title: Article title
        description: Article description
        keywords: Article keywords
        
    Returns:
        float: Sentiment score between -1 and 1
    """
    text = f"{title} {description}".lower()
    keyword_text = " ".join(keywords).lower()
    combined_text = f"{text} {keyword_text}"
    
    # Define positive and negative keywords
    positive_keywords = [
        'positive', 'bullish', 'gain', 'profit', 'growth', 'up', 'rise', 'increase',
        'strong', 'beat', 'exceed', 'outperform', 'buy', 'upgrade', 'positive',
        'earnings', 'revenue', 'sales', 'success', 'win', 'advantage'
    ]
    
    negative_keywords = [
        'negative', 'bearish', 'loss', 'decline', 'down', 'fall', 'decrease',
        'weak', 'miss', 'underperform', 'sell', 'downgrade', 'negative',
        'loss', 'debt', 'risk', 'concern', 'worry', 'problem', 'issue'
    ]
    
    # Count positive and negative keywords
    positive_count = sum(1 for word in positive_keywords if word in combined_text)
    negative_count = sum(1 for word in negative_keywords if word in combined_text)
    
    # Calculate sentiment score
    total_keywords = positive_count + negative_count
    if total_keywords == 0:
        return 0.0
    
    sentiment_score = (positive_count - negative_count) / total_keywords
    return max(min(sentiment_score, 1.0), -1.0)


def _get_sentiment_label(sentiment_score: float) -> str:
    """
    Convert sentiment score to label.
    
    Args:
        sentiment_score: Sentiment score between -1 and 1
        
    Returns:
        str: Sentiment label (positive/negative/neutral)
    """
    if sentiment_score > 0.1:
        return 'positive'
    elif sentiment_score < -0.1:
        return 'negative'
    else:
        return 'neutral'


def get_polygon_news_for_tickers(tickers: List[str], 
                                days_back: Optional[int] = None,
                                time_from: Optional[str] = None,
                                time_to: Optional[str] = None,
                                limit_per_ticker: int = 1000) -> pd.DataFrame:
    """
    Gets Polygon news data for specific tickers over a specified time period.
    
    Args:
        tickers: List of stock ticker symbols
        days_back: Number of days to look back for news (default: 30 if no time_from/time_to provided)
        time_from: Start time for news search (ISO format)
        time_to: End time for news search (ISO format)
        limit_per_ticker: Maximum number of articles per ticker (default: 1000)
        
    Returns:
        pd.DataFrame: Combined news data for all tickers
    """
    if not _validate_polygon_key():
        return pd.DataFrame()
    
    # Calculate time range if not provided
    if time_from is None or time_to is None:
        if days_back is None:
            days_back = 30
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # Strip time info and use 0000 and 2359 respectively
        time_from = start_date.strftime('%Y-%m-%d') + 'T0000Z'
        time_to = end_date.strftime('%Y-%m-%d') + 'T2359Z'
    
    print(f"Downloading Polygon news for {len(tickers)} tickers from {time_from} to {time_to}...")
    if limit_per_ticker > 1000:
        limit_per_ticker = 1000
    all_news = []
    
    for ticker in tickers:
        print(f"  Downloading news for {ticker}...")
        try:
            ticker_news = get_polygon_news_sentiment(
                tickers=[ticker],
                published_utc_gte=time_from,
                published_utc_lte=time_to,
                limit=limit_per_ticker
            )
            
            if not ticker_news.empty:
                all_news.append(ticker_news)
                print(f"    ✅ Found {len(ticker_news)} articles for {ticker}")
            else:
                print(f"    ⚠️  No news found for {ticker}")

            _rate_limit_delay()
                
        except Exception as e:
            print(f"    ❌ Error downloading news for {ticker}: {e}")
    
    if all_news:
        combined_news = pd.concat(all_news, ignore_index=True)
        print(f"✅ Successfully downloaded {len(combined_news)} total articles")
        return combined_news
    else:
        print("❌ No news data collected for any ticker")
        return pd.DataFrame()


def get_polygon_market_news(days_back: int = 7,
                           limit: int = 100) -> pd.DataFrame:
    """
    Gets general market news from Polygon without specific ticker focus.
    
    Args:
        days_back: Number of days to look back for news (default: 7)
        limit: Maximum number of articles to retrieve (default: 100)
        
    Returns:
        pd.DataFrame: Market news data
    """
    if not _validate_polygon_key():
        return pd.DataFrame()
    
    print(f"Downloading Polygon market news for the last {days_back} days...")
    
    # Calculate time range (time-agnostic)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    # Strip time info and use 0000 and 2359 respectively
    published_utc_gte = start_date.strftime('%Y-%m-%d') + 'T0000Z'
    published_utc_lte = end_date.strftime('%Y-%m-%d') + 'T2359Z'
    
    try:
        market_news = get_polygon_news_sentiment(
            published_utc_gte=published_utc_gte,
            published_utc_lte=published_utc_lte,
            limit=limit
        )
        
        if not market_news.empty:
            print(f"✅ Successfully downloaded {len(market_news)} Polygon market news articles")
        else:
            print("⚠️  No Polygon market news found")
            
        return market_news
        
    except Exception as e:
        print(f"❌ Error downloading Polygon market news: {e}")
        return pd.DataFrame()


def analyze_polygon_news_sentiment(news_data: pd.DataFrame) -> Dict:
    """
    Analyzes sentiment distribution in Polygon news data.
    
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


def get_polygon_news_summary(news_data: pd.DataFrame, top_n: int = 5) -> Dict:
    """
    Creates a summary of the most relevant Polygon news articles.
    
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

def _load_master_metadata() -> Dict:
    """
    Loads the master metadata file containing information about all downloaded data.
    
    Returns:
        Dict: Master metadata containing ticker information and date ranges
    """
    if not MASTER_FILE_PATH.exists():
        return {}
    
    try:
        with open(MASTER_FILE_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading master metadata: {e}")
        return {}


def _save_master_metadata(metadata: Dict):
    """
    Saves the master metadata file.
    
    Args:
        metadata: Master metadata dictionary to save
    """
    try:
        POLYGON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MASTER_FILE_PATH, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        print(f"❌ Error saving master metadata: {e}")


def _get_ticker_cache_dir(ticker: str) -> epath.Path:
    """
    Gets the cache directory for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Path: Cache directory path for the ticker
    """
    return POLYGON_CACHE_DIR / ticker


def _get_ticker_metadata_file(ticker: str) -> epath.Path:
    """
    Gets the metadata file path for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Path: Metadata file path for the ticker
    """
    return _get_ticker_cache_dir(ticker) / "metadata.json"


def _load_ticker_metadata(ticker: str) -> Dict:
    """
    Loads metadata for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict: Ticker metadata containing date ranges and file information
    """
    metadata_file = _get_ticker_metadata_file(ticker)
    if not metadata_file.exists():
        return {}
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading metadata for {ticker}: {e}")
        return {}


def _save_ticker_metadata(ticker: str, metadata: Dict):
    """
    Saves metadata for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        metadata: Metadata dictionary to save
    """
    try:
        cache_dir = _get_ticker_cache_dir(ticker)
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = _get_ticker_metadata_file(ticker)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        print(f"❌ Error saving metadata for {ticker}: {e}")


def _get_ticker_data_file(ticker: str) -> epath.Path:
    """
    Gets the data file path for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Path: Data file path for the ticker
    """
    return _get_ticker_cache_dir(ticker) / "news_data.pkl"


def _is_cache_fresh(ticker: str, incremental: Optional[bool] = None) -> bool:
    """
    Checks if the cached data for a ticker exists and is valid.
    Historical data is always considered valid once downloaded.
    
    Args:
        ticker: Stock ticker symbol
        incremental: For check if update is required
        
    Returns:
        bool: True if cache exists and is valid, False otherwise
    """
    metadata = _load_ticker_metadata(ticker)
    if not metadata or 'last_updated' not in metadata:
        return False
    
    # Check if the data file actually exists
    data_file = _get_ticker_data_file(ticker)
    if not data_file.exists():
        return False
    
    # If incremental is True, check if last_updated is before today
    if incremental:
        try:
            last_updated = pd.to_datetime(metadata['last_updated'])
            today = pd.Timestamp.now().normalize()
            if last_updated.date() < today.date():
                return False
        except Exception:
            return False
    
    # Historical data is always valid once downloaded
    # Only check if the cache exists and has valid metadata
    try:
        # Verify the metadata is valid
        if 'total_articles' in metadata and metadata['total_articles'] > 0:
            return True
        else:
            return False
    except Exception:
        return False


def download_all_polygon_news_after_date(tickers: List[str], 
                                        start_date: str,
                                        end_date: Optional[str] = None,
                                        force_download: bool = False) -> Dict:
    """
    Downloads all Polygon news data for specified tickers after a given date.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for news download (ISO format)
        end_date: End date for news download (ISO format, defaults to now)
        force_download: Force re-download even if cache is fresh
        
    Returns:
        Dict: Summary of download results for each ticker
    """
    if not _validate_polygon_key():
        return {}
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading all Polygon news data after {start_date} for {len(tickers)} tickers...")
    
    results = {}
    master_metadata = _load_master_metadata()
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # Check if we need to download new data
        if not force_download and _is_cache_fresh(ticker):
            print(f"  ✅ Cache is fresh for {ticker}, skipping download")
            ticker_metadata = _load_ticker_metadata(ticker)
            master_metadata[ticker] = {
                'last_updated': ticker_metadata['last_updated'],
                'start_date': ticker_metadata['start_date'],
                'end_date': ticker_metadata['end_date'],
                'total_articles': ticker_metadata['total_articles'],
                'date_range': ticker_metadata['date_range']
            }
            results[ticker] = {'status': 'cached', 'articles': ticker_metadata['total_articles']}
            continue
        
        try:
            # Download news data for this ticker
            news_data = get_polygon_news_sentiment(
                tickers=[ticker],
                published_utc_gte=start_date,
                published_utc_lte=end_date,
                limit=1000  # Large limit to get all data
            )
            
            if not news_data.empty:
                # Save data to ticker-specific cache
                cache_dir = _get_ticker_cache_dir(ticker)
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                data_file = _get_ticker_data_file(ticker)
                news_data.to_pickle(data_file)
                
                # Create metadata for this ticker
                ticker_metadata = {
                    'ticker': ticker,
                    'start_date': start_date,
                    'end_date': end_date,
                    'download_date': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'total_articles': len(news_data),
                    'data_file': str(data_file),
                    'date_range': {
                        'earliest': str(news_data['time_published'].min()) if 'time_published' in news_data.columns else '',
                        'latest': str(news_data['time_published'].max()) if 'time_published' in news_data.columns else ''
                    }
                }
                
                # Save ticker metadata
                _save_ticker_metadata(ticker, ticker_metadata)
                
                # Update master metadata
                master_metadata[ticker] = {
                    'last_updated': datetime.now().isoformat(),
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_articles': len(news_data),
                    'date_range': ticker_metadata['date_range']
                }
                
                print(f"  ✅ Downloaded {len(news_data)} articles for {ticker}")
                results[ticker] = {'status': 'downloaded', 'articles': len(news_data)}
                
            else:
                print(f"  ⚠️  No news data found for {ticker}")
                results[ticker] = {'status': 'no_data', 'articles': 0}
            _rate_limit_delay()    
        except Exception as e:
            print(f"  ❌ Error downloading data for {ticker}: {e}")
            results[ticker] = {'status': 'error', 'articles': 0, 'error': str(e)}
    
    # Save updated master metadata
    _save_master_metadata(master_metadata)
    
    print(f"\n✅ Download complete. Summary:")
    for ticker, result in results.items():
        print(f"  {ticker}: {result['status']} ({result['articles']} articles)")
    
    return results


def download_incremental_polygon_news(tickers: List[str], 
                                    days_back: int = 7,
                                    force_download: bool = False) -> Dict:
    """
    Downloads incremental Polygon news data for specified tickers.
    Only downloads data that's newer than what's already cached.
    
    Args:
        tickers: List of stock ticker symbols
        days_back: Number of days to look back for new data
        force_download: Force re-download even if cache is fresh
        
    Returns:
        Dict: Summary of incremental download results
    """
    if not _validate_polygon_key():
        return {}
    
    print(f"Downloading incremental Polygon news data for {len(tickers)} tickers...")
    
    results = {}
    master_metadata = _load_master_metadata()
    
    # Calculate date range for incremental download
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # Load existing metadata
        ticker_metadata = _load_ticker_metadata(ticker)

        master_metadata[ticker] = {
            'last_updated': ticker_metadata['last_updated'],
            'start_date': ticker_metadata['start_date'],
            'end_date': ticker_metadata['end_date'],
            'total_articles': ticker_metadata['total_articles'],
            'date_range': ticker_metadata['date_range']
        }
        
        # Determine the actual start date for this ticker
        if ticker_metadata and 'date_range' in ticker_metadata and 'latest' in ticker_metadata['date_range']:
            try:
                last_article_date = pd.to_datetime(ticker_metadata['date_range']['latest'])
                # Start from 1 day after the last article to avoid duplicates
                incremental_start = last_article_date + timedelta(days=1)
                if incremental_start < end_date:
                    actual_start_date = incremental_start.strftime('%Y-%m-%d')
                    print(f"  📅 Incremental download from {actual_start_date} to {end_date_str}")
                else:
                    print(f"  ✅ No new data needed for {ticker}")
                    results[ticker] = {'status': 'up_to_date', 'articles': 0}
                    continue
            except Exception:
                actual_start_date = start_date_str
                print(f"  📅 Full download from {actual_start_date} to {end_date_str}")
        else:
            actual_start_date = start_date_str
            print(f"  📅 Initial download from {actual_start_date} to {end_date_str}")
        
        # Check if we need to download
        if not force_download and _is_cache_fresh(ticker, incremental=True):
            print(f"  ✅ Cache is fresh for {ticker}, skipping download")
            results[ticker] = {'status': 'cached', 'articles': ticker_metadata['total_articles']}
            continue
        
        try:
            # Download new data
            new_data = get_polygon_news_sentiment(
                tickers=[ticker],
                published_utc_gte=actual_start_date,
                published_utc_lte=end_date_str,
                limit=10000
            )
            
            if not new_data.empty:
                # Load existing data if available
                existing_data = pd.DataFrame()
                data_file = _get_ticker_data_file(ticker)
                if data_file.exists():
                    try:
                        existing_data = pd.read_pickle(data_file)
                    except Exception:
                        pass
                
                # Combine existing and new data
                if not existing_data.empty:
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                    # Remove duplicates based on title and time_published
                    combined_data = combined_data.drop_duplicates(subset=['title', 'time_published'], keep='first')
                    combined_data = combined_data.sort_values('time_published', ascending=False).reset_index(drop=True)
                else:
                    combined_data = new_data
                
                # Save combined data
                cache_dir = _get_ticker_cache_dir(ticker)
                cache_dir.mkdir(parents=True, exist_ok=True)
                combined_data.to_pickle(data_file)
                
                # Update metadata
                ticker_metadata = {
                    'ticker': ticker,
                    'start_date': ticker_metadata.get('start_date', actual_start_date),
                    'end_date': end_date_str,
                    'download_date': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'total_articles': len(combined_data),
                    'new_articles': len(new_data),
                    'data_file': str(data_file),
                    'date_range': {
                        'earliest': str(combined_data['time_published'].min()) if 'time_published' in combined_data.columns else '',
                        'latest': str(combined_data['time_published'].max()) if 'time_published' in combined_data.columns else ''
                    }
                }
                
                _save_ticker_metadata(ticker, ticker_metadata)
                
                # Update master metadata
                master_metadata[ticker] = {
                    'last_updated': datetime.now().isoformat(),
                    'start_date': ticker_metadata['start_date'],
                    'end_date': end_date_str,
                    'total_articles': len(combined_data),
                    'new_articles': len(new_data),
                    'date_range': ticker_metadata['date_range']
                }
                
                print(f"  ✅ Downloaded {len(new_data)} new articles for {ticker} (total: {len(combined_data)})")
                results[ticker] = {'status': 'incremental', 'articles': len(new_data), 'total': len(combined_data)}
                
            else:
                print(f"  ⚠️  No new data found for {ticker}")
                results[ticker] = {'status': 'no_new_data', 'articles': 0}
                
        except Exception as e:
            print(f"  ❌ Error downloading incremental data for {ticker}: {e}")
            results[ticker] = {'status': 'error', 'articles': 0, 'error': str(e)}
    
    # Save updated master metadata
    _save_master_metadata(master_metadata)
    
    print(f"\n✅ Incremental download complete. Summary:")
    for ticker, result in results.items():
        print(f"  {ticker}: {result['status']} ({result['articles']} new articles)")
    
    return results


def get_cached_polygon_news(ticker: str) -> pd.DataFrame:
    """
    Retrieves cached Polygon news data for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        pd.DataFrame: Cached news data for the ticker
    """
    data_file = _get_ticker_data_file(ticker)
    if not data_file.exists():
        print(f"❌ No cached data found for {ticker}")
        return pd.DataFrame()
    
    try:
        data = pd.read_pickle(data_file)
        print(f"✅ Loaded {len(data)} cached articles for {ticker}")
        return data
    except Exception as e:
        print(f"❌ Error loading cached data for {ticker}: {e}")
        return pd.DataFrame()


def get_polygon_news_from_cache(tickers: List[str], 
                               days_back: Optional[int] = None,
                               time_from: Optional[str] = None,
                               time_to: Optional[str] = None,
                               limit_per_ticker: int = 1000,
                               use_cached_data: Optional[bool] = None) -> pd.DataFrame:
    """
    Enhanced version of get_polygon_news_for_tickers that uses cached data.
    
    Args:
        tickers: List of stock ticker symbols
        days_back: Number of days to look back for news (default: 30 if no time_from/time_to provided)
        time_from: Start time for news search (ISO format)
        time_to: End time for news search (ISO format)
        limit_per_ticker: Maximum number of articles per ticker (default: 1000)
        use_cached_data: Override cached data setting (uses config if None)
        
    Returns:
        pd.DataFrame: News data from cache if available
    """
    if use_cached_data is None:
        use_cached_data = USE_CACHED_DATA
    
    # Calculate time range if not provided (matching get_polygon_news_for_tickers logic)
    if time_from is None or time_to is None:
        if days_back is None:
            days_back = 30
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # Strip time info and use 0000 and 2359 respectively
        time_from = start_date.strftime('%Y-%m-%d') + 'T0000Z'
        time_to = end_date.strftime('%Y-%m-%d') + 'T2359Z'
    
    # If cached data is enabled and we have specific tickers, try to get cached data first
    if use_cached_data and tickers:
        print(f"Using cached data for {len(tickers)} tickers...")
        
        # Try to get data from cache first
        cached_data = []
        missing_tickers = []
        
        for ticker in tickers:
            ticker_data = get_cached_polygon_news(ticker)
            if not ticker_data.empty:
                cached_data.append(ticker_data)
            else:
                missing_tickers.append(ticker)
        
        # If we have missing tickers, download them
        if missing_tickers:
            raise RuntimeError(
                f"Missing cached Polygon news data for tickers: {missing_tickers}. "
                "Auto-download is disabled for this operation. Please download the data first."
            )
        
        # Combine all cached data
        if cached_data:
            combined_data = pd.concat(cached_data, ignore_index=True)
            
            # Filter by date range if specified
            if time_from or time_to:
                if time_from:
                    combined_data = combined_data[
                        combined_data['time_published'] >= pd.to_datetime(time_from)
                    ]
                if time_to:
                    combined_data = combined_data[
                        combined_data['time_published'] <= pd.to_datetime(time_to)
                    ]
            
            print(f"✅ Retrieved {len(combined_data)} articles from cache")
            return combined_data
    
    # Fall back to original function if cached data is disabled or no specific tickers
    return get_polygon_news_for_tickers(
        tickers=tickers,
        days_back=days_back,
        time_from=time_from,
        time_to=time_to,
        limit_per_ticker=limit_per_ticker
    ) 