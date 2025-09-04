#!/usr/bin/env python3
"""
Test script for the new time_from and time_to parameters in get_news_for_tickers.

This script demonstrates how to use specific start and end times instead of days_back.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.news_config import NewsProvider
from src.data.unified_news_data import (
    get_news_for_tickers,
    set_news_provider,
    get_current_provider
)
from datetime import datetime, timedelta


def test_time_parameters():
    """Test the new time_from and time_to parameters."""
    print("🧪 Testing Time Parameters for get_news_for_tickers")
    print("=" * 60)
    
    test_tickers = ['AAPL', 'MSFT']
    
    # Test 1: Using days_back (original method)
    print("\n1. Testing with days_back parameter...")
    try:
        news_data = get_news_for_tickers(
            tickers=test_tickers,
            days_back=7,
            limit_per_ticker=1000
        )
        print(f"   ✅ Retrieved {len(news_data)} articles using days_back=7")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Using time_from and time_to (new method)
    print("\n2. Testing with time_from and time_to parameters...")
    try:
        # Calculate specific time range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Format for Alpha Vantage (YYYYMMDDTHHMM)
        time_from = start_date.strftime('%Y%m%dT0000')
        time_to = end_date.strftime('%Y%m%dT2359')
        
        news_data = get_news_for_tickers(
            tickers=test_tickers,
            time_from=time_from,
            time_to=time_to,
            limit_per_ticker=1000
        )
        print(f"   ✅ Retrieved {len(news_data)} articles using time_from={time_from}, time_to={time_to}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Using time_from and time_to with Polygon provider
    print("\n3. Testing with Polygon provider and time parameters...")
    try:
        set_news_provider(NewsProvider.POLYGON)
        print(f"   Using provider: {get_current_provider().value}")
        
        # Format for Polygon (ISO format)
        time_from_iso = start_date.strftime('%Y-%m-%d') + 'T0000Z'
        time_to_iso = end_date.strftime('%Y-%m-%d') + 'T2359Z'
        
        news_data = get_news_for_tickers(
            tickers=test_tickers,
            time_from=time_from_iso,
            time_to=time_to_iso,
            limit_per_ticker=1000
        )
        print(f"   ✅ Retrieved {len(news_data)} articles using Polygon with time parameters")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Using only time_from (should use current time as end)
    print("\n4. Testing with only time_from parameter...")
    try:
        set_news_provider(NewsProvider.ALPHA_VANTAGE)
        print(f"   Using provider: {get_current_provider().value}")
        
        # Use only start time
        news_data = get_news_for_tickers(
            tickers=test_tickers,
            time_from=time_from,
            limit_per_ticker=1000
        )
        print(f"   ✅ Retrieved {len(news_data)} articles using only time_from")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Time parameter testing completed!")


def demonstrate_usage():
    """Demonstrate different usage patterns."""
    print("\n📋 Usage Examples:")
    print("=" * 30)
    
    print("\n1. Using days_back (original method):")
    print("   news = get_news_for_tickers(['AAPL', 'MSFT'], days_back=30)")
    
    print("\n2. Using specific time range (Alpha Vantage format):")
    print("   news = get_news_for_tickers(['AAPL', 'MSFT'],")
    print("                                 time_from='20240101T0000',")
    print("                                 time_to='20240131T2359')")
    
    print("\n3. Using specific time range (Polygon format):")
    print("   set_news_provider(NewsProvider.POLYGON)")
    print("   news = get_news_for_tickers(['AAPL', 'MSFT'],")
    print("                                 time_from='2024-01-01T0000Z',")
    print("                                 time_to='2024-01-31T2359Z')")
    
    print("\n4. Using only start time (end defaults to now):")
    print("   news = get_news_for_tickers(['AAPL', 'MSFT'],")
    print("                                 time_from='20240101T0000')")
    
    print("\n5. Using only end time (start defaults to days_back):")
    print("   news = get_news_for_tickers(['AAPL', 'MSFT'],")
    print("                                 time_to='20240131T2359')")
    
    print("\n6. Using custom limit (default is 1000):")
    print("   news = get_news_for_tickers(['AAPL', 'MSFT'],")
    print("                                 limit_per_ticker=500)")


if __name__ == "__main__":
    print("🚀 Starting Time Parameter Tests")
    print("=" * 60)
    
    try:
        test_time_parameters()
        demonstrate_usage()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 