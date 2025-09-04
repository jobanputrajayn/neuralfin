#!/usr/bin/env python3
"""
Test script for Polygon news data functionality.

This script demonstrates how to use the new Polygon-based news data source
and compare it with the existing Alpha Vantage implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.news_config import NewsProvider
from src.data.unified_news_data import set_news_provider
from src.data.unified_news_data import (
    get_news_for_tickers,
    get_market_news,
    analyze_news_sentiment,
    get_news_summary,
    compare_providers,
    validate_provider_config
)
import pandas as pd


def test_polygon_news():
    """Test Polygon news data functionality."""
    print("🧪 Testing Polygon News Data Functionality")
    print("=" * 50)
    
    # Test tickers
    test_tickers = ['AAPL', 'MSFT']
    
    # Test provider validation
    print("\n1. Testing provider validation...")
    av_valid = validate_provider_config(NewsProvider.ALPHA_VANTAGE)
    poly_valid = validate_provider_config(NewsProvider.POLYGON)
    
    print(f"   Alpha Vantage: {'✅ Valid' if av_valid else '❌ Invalid'}")
    print(f"   Polygon: {'✅ Valid' if poly_valid else '❌ Invalid'}")
    
    # # Compare providers
    # print("\n2. Comparing providers...")
    # comparison = compare_providers(test_tickers, days_back=7, limit=10)
    
    # print(f"   Alpha Vantage articles: {comparison['alpha_vantage']['articles_count']}")
    # print(f"   Polygon articles: {comparison['polygon']['articles_count']}")
    # print(f"   Recommended provider: {comparison['summary']['recommended_provider']}")
    
    # Test Polygon-specific functionality
    print("\n3. Testing Polygon news data...")
    set_news_provider(NewsProvider.POLYGON)
    
    # Get news for specific tickers
    print("   Fetching news for AAPL and MSFT...")
    news_data = get_news_for_tickers(test_tickers, days_back=7, limit_per_ticker=5)
    
    if not news_data.empty:
        print(f"   ✅ Retrieved {len(news_data)} articles")
        
        # Analyze sentiment
        print("   Analyzing sentiment...")
        sentiment_analysis = analyze_news_sentiment(news_data)
        
        if sentiment_analysis:
            print("   Sentiment distribution:")
            for label, count in sentiment_analysis.get('sentiment_distribution', {}).items():
                print(f"     {label}: {count}")
        
        # Get summary
        print("   Getting news summary...")
        summary = get_news_summary(news_data, top_n=3)
        
        if summary.get('top_articles'):
            print("   Top articles:")
            for i, article in enumerate(summary['top_articles'][:2], 1):
                print(f"     {i}. {article['title'][:60]}...")
                print(f"        Sentiment: {article['sentiment_label']} ({article['sentiment_score']:.2f})")
    else:
        print("   ⚠️  No news data retrieved")
    
    # Test market news
    print("\n4. Testing market news...")
    market_news = get_market_news(days_back=3, limit=20)
    
    if not market_news.empty:
        print(f"   ✅ Retrieved {len(market_news)} market news articles")
    else:
        print("   ⚠️  No market news retrieved")
    
    print("\n✅ Polygon news testing completed!")


def test_provider_switching():
    """Test switching between news providers."""
    print("\n🔄 Testing Provider Switching")
    print("=" * 30)
    
    test_tickers = ['TSLA']
    
    # Test Alpha Vantage
    print("\n1. Testing Alpha Vantage...")
    set_news_provider(NewsProvider.ALPHA_VANTAGE)
    av_news = get_news_for_tickers(test_tickers, days_back=3, limit_per_ticker=3)
    print(f"   Alpha Vantage articles: {len(av_news)}")
    
    # Test Polygon
    print("\n2. Testing Polygon...")
    set_news_provider(NewsProvider.POLYGON)
    poly_news = get_news_for_tickers(test_tickers, days_back=3, limit_per_ticker=3)
    print(f"   Polygon articles: {len(poly_news)}")
    
    # Compare data formats
    if not av_news.empty and not poly_news.empty:
        print("\n3. Comparing data formats...")
        print(f"   Alpha Vantage columns: {list(av_news.columns)}")
        print(f"   Polygon columns: {list(poly_news.columns)}")
        
        # Check if formats match
        av_cols = set(av_news.columns)
        poly_cols = set(poly_news.columns)
        common_cols = av_cols.intersection(poly_cols)
        
        print(f"   Common columns: {len(common_cols)}")
        print(f"   Format compatibility: {'✅ Good' if len(common_cols) >= 8 else '⚠️  Partial'}")


if __name__ == "__main__":
    print("🚀 Starting Polygon News Data Tests")
    print("=" * 50)
    
    try:
        test_polygon_news()
        #test_provider_switching()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Usage Examples:")
        print("   # Set Polygon as default provider")
        print("   from src.config.news_config import NewsProvider")
        print("   from src.data.unified_news_data import set_news_provider")
        print("   set_news_provider(NewsProvider.POLYGON)")
        print("")
        print("   # Get news data (uses configured provider)")
        print("   from src.data.unified_news_data import get_news_for_tickers")
        print("   news = get_news_for_tickers(['AAPL', 'MSFT'], days_back=7)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 