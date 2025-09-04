"""
News data source configuration for the Hyper framework.

Allows selection between different news data providers (Alpha Vantage, Polygon.io).
"""

import os
from enum import Enum
from typing import Optional


class NewsProvider(Enum):
    """Available news data providers."""
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"


class NewsConfig:
    """Configuration for news data sources."""
    
    def __init__(self, provider: NewsProvider = NewsProvider.ALPHA_VANTAGE):
        self.provider = provider
        self._alpha_vantage_key = None
        self._polygon_key = None
    
    @property
    def alpha_vantage_key(self) -> Optional[str]:
        """Get Alpha Vantage API key."""
        if self._alpha_vantage_key is None:
            try:
                from .av_key import ALPHA_VANTAGE_API_KEY
                self._alpha_vantage_key = ALPHA_VANTAGE_API_KEY
            except ImportError:
                self._alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        return self._alpha_vantage_key
    
    @property
    def polygon_key(self) -> Optional[str]:
        """Get Polygon API key."""
        if self._polygon_key is None:
            try:
                from .polygon_key import POLYGON_API_KEY
                self._polygon_key = POLYGON_API_KEY
            except ImportError:
                self._polygon_key = os.getenv('POLYGON_API_KEY')
        return self._polygon_key
    
    def get_api_key(self) -> Optional[str]:
        """Get API key for the current provider."""
        if self.provider == NewsProvider.ALPHA_VANTAGE:
            return self.alpha_vantage_key
        elif self.provider == NewsProvider.POLYGON:
            return self.polygon_key
        return None
    
    def validate_api_key(self) -> bool:
        """Validate that API key is available for current provider."""
        api_key = self.get_api_key()
        if not api_key or api_key == "YOUR_POLYGON_API_KEY_HERE":
            return False
        return True


# Default configuration
DEFAULT_NEWS_PROVIDER = NewsProvider.POLYGON
news_config = NewsConfig(DEFAULT_NEWS_PROVIDER) 