"""
Data package for the Hyper framework.

Contains data loading, preprocessing, and sequence generation components.
"""

# Import stock data functions
from .stock_data import get_stock_data, get_large_cap_tickers

# Import news data functions
from .news_data import (
    get_news_sentiment,
    get_news_for_tickers,
    get_market_news,
    analyze_news_sentiment,
    get_news_summary
)

# Import sequence generator with error handling
try:
    from .sequence_generator import StockSequenceGenerator, PrefetchGenerator
except ImportError as e:
    # If relative import fails, try absolute import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.sequence_generator import StockSequenceGenerator, PrefetchGenerator

__all__ = [
    'get_stock_data',
    'get_large_cap_tickers',
    'get_news_sentiment',
    'get_news_for_tickers',
    'get_market_news',
    'analyze_news_sentiment',
    'get_news_summary',
    'StockSequenceGenerator',
    'PrefetchGenerator'
] 