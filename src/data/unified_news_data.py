"""
Unified news data interface for the Hyper framework.

Provides a unified interface for accessing news data from multiple providers
(Alpha Vantage, Polygon.io) with automatic provider selection.
"""

import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta

from src.config.news_config import NewsConfig, NewsProvider, news_config
from src.data import news_data as alpha_vantage_news
from src.data import polygon_news_data as polygon_news


def get_news_sentiment(tickers: Optional[List[str]] = None, 
                      topics: Optional[List[str]] = None,
                      time_from: Optional[str] = None,
                      time_to: Optional[str] = None,
                      limit: int = 50,
                      provider: Optional[NewsProvider] = None) -> pd.DataFrame:
    """
    Downloads news sentiment data from the configured provider.
    
    Args:
        tickers (list, optional): List of stock ticker symbols to search for.
        topics (list, optional): List of news topics to search for (Alpha Vantage only).
        time_from (str, optional): Start time for news search.
        time_to (str, optional): End time for news search.
        limit (int): Maximum number of news articles to retrieve (default: 50).
        provider (NewsProvider, optional): Override default provider selection.
        
    Returns:
        pd.DataFrame: DataFrame containing news sentiment data
    """
    # Determine which provider to use
    if provider is None:
        provider = news_config.provider
    
    print(f"Using news provider: {provider.value}")
    
    if provider == NewsProvider.ALPHA_VANTAGE:
        return alpha_vantage_news.get_news_sentiment(
            tickers=tickers,
            topics=topics,
            time_from=time_from,
            time_to=time_to,
            limit=limit
        )
    elif provider == NewsProvider.POLYGON:
        # Convert Alpha Vantage time format to Polygon format if needed
        published_utc_gte = _convert_time_format(time_from) if time_from else None
        published_utc_lte = _convert_time_format(time_to) if time_to else None
        
        return polygon_news.get_polygon_news_sentiment(
            tickers=tickers,
            published_utc_gte=published_utc_gte,
            published_utc_lte=published_utc_lte,
            limit=1000
        )
    else:
        print(f"❌ Unsupported news provider: {provider}")
        return pd.DataFrame()


def get_news_for_tickers(tickers: List[str], 
                        days_back: Optional[int] = None,
                        time_from: Optional[str] = None,
                        time_to: Optional[str] = None,
                        limit_per_ticker: int = 10,
                        provider: Optional[NewsProvider] = None) -> pd.DataFrame:
    """
    Gets news data for specific tickers over a specified time period.
    
    Args:
        tickers: List of stock ticker symbols
        days_back: Number of days to look back for news (default: 30 if no time_from/time_to provided)
        time_from: Start time for news search (format depends on provider)
        time_to: End time for news search (format depends on provider)
        limit_per_ticker: Maximum number of articles per ticker (default: 10)
        provider (NewsProvider, optional): Override default provider selection.
        
    Returns:
        pd.DataFrame: Combined news data for all tickers
    """
    if provider is None:
        provider = news_config.provider
    
    print(f"Using news provider: {provider.value}")
    
    if provider == NewsProvider.ALPHA_VANTAGE:
        return alpha_vantage_news.get_news_for_tickers(
            tickers=tickers,
            days_back=days_back,
            time_from=time_from,
            time_to=time_to,
            limit_per_ticker=limit_per_ticker
        )
    elif provider == NewsProvider.POLYGON:
        # Convert time format for Polygon if provided
        polygon_time_from = None
        polygon_time_to = None
        
        if time_from:
            try:
                # Convert to Polygon format (ISO with Z)
                dt_from = pd.to_datetime(time_from)
                polygon_time_from = dt_from.strftime('%Y-%m-%d') + 'T00:00:00Z'
            except Exception:
                polygon_time_from = time_from
        
        if time_to:
            try:
                # Convert to Polygon format (ISO with Z)
                dt_to = pd.to_datetime(time_to)
                polygon_time_to = dt_to.strftime('%Y-%m-%d') + 'T23:59:59Z'
            except Exception:
                polygon_time_to = time_to
        
        return polygon_news.get_polygon_news_from_cache(
            tickers=tickers,
            days_back=days_back,
            time_from=polygon_time_from,
            time_to=polygon_time_to,
            limit_per_ticker=limit_per_ticker
        )
    else:
        print(f"❌ Unsupported news provider: {provider}")
        return pd.DataFrame()


def get_market_news(topics: Optional[List[str]] = None,
                   days_back: int = 7,
                   limit: int = 100,
                   provider: Optional[NewsProvider] = None) -> pd.DataFrame:
    """
    Gets general market news without specific ticker focus.
    
    Args:
        topics: List of news topics (Alpha Vantage only)
        days_back: Number of days to look back for news (default: 7)
        limit: Maximum number of articles to retrieve (default: 100)
        provider (NewsProvider, optional): Override default provider selection.
        
    Returns:
        pd.DataFrame: Market news data
    """
    if provider is None:
        provider = news_config.provider
    
    print(f"Using news provider: {provider.value}")
    
    if provider == NewsProvider.ALPHA_VANTAGE:
        return alpha_vantage_news.get_market_news(
            topics=topics,
            days_back=days_back,
            limit=limit
        )
    elif provider == NewsProvider.POLYGON:
        return polygon_news.get_polygon_market_news(
            days_back=days_back,
            limit=limit
        )
    else:
        print(f"❌ Unsupported news provider: {provider}")
        return pd.DataFrame()


def analyze_news_sentiment(news_data: pd.DataFrame, 
                          provider: Optional[NewsProvider] = None) -> Dict:
    """
    Analyzes sentiment distribution in news data.
    
    Args:
        news_data: DataFrame containing news sentiment data
        provider (NewsProvider, optional): Provider for analysis functions.
        
    Returns:
        Dict: Analysis results including sentiment distribution and statistics
    """
    if provider is None:
        provider = news_config.provider
    
    if provider == NewsProvider.ALPHA_VANTAGE:
        return alpha_vantage_news.analyze_news_sentiment(news_data)
    elif provider == NewsProvider.POLYGON:
        return polygon_news.analyze_polygon_news_sentiment(news_data)
    else:
        print(f"❌ Unsupported news provider: {provider}")
        return {}


def get_news_summary(news_data: pd.DataFrame, 
                    top_n: int = 5,
                    provider: Optional[NewsProvider] = None) -> Dict:
    """
    Creates a summary of the most relevant news articles.
    
    Args:
        news_data: DataFrame containing news data
        top_n: Number of top articles to include in summary
        provider (NewsProvider, optional): Provider for summary functions.
        
    Returns:
        Dict: Summary of top news articles
    """
    if provider is None:
        provider = news_config.provider
    
    if provider == NewsProvider.ALPHA_VANTAGE:
        return alpha_vantage_news.get_news_summary(news_data, top_n)
    elif provider == NewsProvider.POLYGON:
        return polygon_news.get_polygon_news_summary(news_data, top_n)
    else:
        print(f"❌ Unsupported news provider: {provider}")
        return {}


def set_news_provider(provider: NewsProvider):
    """
    Set the default news provider for all news data operations.
    
    Args:
        provider: NewsProvider enum value
    """
    global news_config
    news_config.provider = provider
    print(f"✅ News provider set to: {provider.value}")


def get_current_provider() -> NewsProvider:
    """
    Get the currently configured news provider.
    
    Returns:
        NewsProvider: Current provider
    """
    return news_config.provider


def validate_provider_config(provider: Optional[NewsProvider] = None) -> bool:
    """
    Validate that the specified provider is properly configured.
    
    Args:
        provider: NewsProvider to validate (uses current if None)
        
    Returns:
        bool: True if provider is properly configured
    """
    if provider is None:
        provider = news_config.provider
    
    if provider == NewsProvider.ALPHA_VANTAGE:
        return alpha_vantage_news._validate_alpha_vantage_key()
    elif provider == NewsProvider.POLYGON:
        return polygon_news._validate_polygon_key()
    else:
        print(f"❌ Unsupported news provider: {provider}")
        return False


def _convert_time_format(time_str: Optional[str]) -> Optional[str]:
    """
    Convert Alpha Vantage time format to Polygon format.
    
    Args:
        time_str: Time string in Alpha Vantage format (YYYYMMDDTHHMM)
        
    Returns:
        Optional[str]: Time string in Polygon format (ISO format) or None
    """
    if not time_str:
        return None
    
    try:
        # Parse Alpha Vantage format: YYYYMMDDTHHMM
        if 'T' in time_str:
            # Already in ISO-like format, just add Z
            return time_str + 'Z'
        else:
            # Convert from YYYYMMDD format
            dt = datetime.strptime(time_str, '%Y%m%d')
            return dt.isoformat() + 'Z'
    except Exception:
        # If parsing fails, return as-is
        return time_str


def compare_providers(tickers: List[str], 
                     days_back: int = 7,
                     limit: int = 50) -> Dict:
    """
    Compare news data from both providers for the same parameters.
    
    Args:
        tickers: List of ticker symbols to compare
        days_back: Number of days to look back
        limit: Maximum articles per provider
        
    Returns:
        Dict: Comparison results
    """
    comparison = {
        'alpha_vantage': {},
        'polygon': {},
        'summary': {}
    }
    
    # Test Alpha Vantage
    print("Testing Alpha Vantage provider...")
    try:
        set_news_provider(NewsProvider.ALPHA_VANTAGE)
        av_data = get_news_for_tickers(tickers, days_back=days_back, limit_per_ticker=limit)
        comparison['alpha_vantage'] = {
            'articles_count': len(av_data),
            'success': not av_data.empty,
            'sample_articles': av_data.head(3).to_dict('records') if not av_data.empty else []
        }
    except Exception as e:
        comparison['alpha_vantage'] = {
            'articles_count': 0,
            'success': False,
            'error': str(e)
        }
    
    # Test Polygon
    print("Testing Polygon provider...")
    try:
        set_news_provider(NewsProvider.POLYGON)
        poly_data = get_news_for_tickers(tickers, days_back=days_back, limit_per_ticker=limit)
        comparison['polygon'] = {
            'articles_count': len(poly_data),
            'success': not poly_data.empty,
            'sample_articles': poly_data.head(3).to_dict('records') if not poly_data.empty else []
        }
    except Exception as e:
        comparison['polygon'] = {
            'articles_count': 0,
            'success': False,
            'error': str(e)
        }
    
    # Summary
    comparison['summary'] = {
        'alpha_vantage_working': comparison['alpha_vantage'].get('success', False),
        'polygon_working': comparison['polygon'].get('success', False),
        'recommended_provider': 'polygon' if comparison['polygon'].get('success', False) else 'alpha_vantage'
    }
    
    return comparison 