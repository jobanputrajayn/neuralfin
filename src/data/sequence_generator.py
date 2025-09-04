"""
Stock sequence generator for JAX GPT stock predictor.
Simplified version that caches processed data (timestamp, ticker, close price, return, label) 
for each time window, then uses this cached data to generate sequences on demand.
"""

import time
import random
import threading
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import jax
import jax.numpy as jnp
from functools import partial
import queue
import hashlib
import pickle
from pathlib import Path
from etils import epath
from flax.jax_utils import prefetch_to_device

from config.news_config import NewsConfig, NewsProvider

# Import constants from models module using absolute imports
try:
    from models.constants import ACTION_HOLD, ACTION_BUY_CALL, ACTION_BUY_PUT, NUM_CLASSES
    from utils.gpu_utils import get_gpu_manager, get_batch_optimizer
except ImportError:
    # Fallback for relative imports if absolute fails
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.constants import ACTION_HOLD, ACTION_BUY_CALL, ACTION_BUY_PUT, NUM_CLASSES
    from utils.gpu_utils import get_gpu_manager, get_batch_optimizer

# Data cache directory
DATA_CACHE_DIR = epath.Path(Path.cwd() / 'data_cache')


def _get_cache_filename(tickers, time_window, threshold_percent):
    """Generate a unique cache filename based on parameters."""
    # Create a consistent string representation of the parameters
    ticker_string = "_".join(sorted(tickers))
    param_string = f"{ticker_string}_{time_window}_{threshold_percent}"
    # Use SHA256 hash to create a short, unique filename
    hash_object = hashlib.sha256(param_string.encode())
    return DATA_CACHE_DIR / f"processed_data_{hash_object.hexdigest()}.pkl"


class StockSequenceGenerator:
    """
    Generates batches of stock sequences (X), labels (Y), and returns (R) on demand.
    This avoids loading the entire dataset into memory.
    
    Features:
    - Adaptive batch sizing for optimal performance
    - Memory-aware processing with JAX optimization
    - Prefetching support for GPU optimization
    - Caching for improved efficiency
    - Optimized data transfers for maximum GPU utilization
    """
    def __init__(self, sequence_indices_to_use: list, # List of start_idx integers
                 all_ohlcv_data: pd.DataFrame,
                 seq_length: int, time_window: int,
                 scaler_mean: float, scaler_std: float,
                 batch_size: int, shuffle_indices: bool = True,
                 tickers: Optional[list] = None, enable_caching: bool = False,
                 cache_size: int = 1000, adaptive_batch_size: bool = False,
                 target_batch_time: float = 0.05,
                 # New JAX-aware parameters
                 jax_memory_fraction: float = 0.8,
                 jax_recompilation_threshold: int = 10,
                 jax_compile_warmup_batches: int = 3,
                 jax_max_batch_size: int = 256,
                 jax_min_batch_size: int = 1,
                 jax_memory_monitoring: bool = True,
                 # Enhanced features parameters
                 include_news: bool = False,
                 include_text: bool = False,
                 news_window: int = 7,
                 text_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 news_cache_dir: str = 'news_cache',
                 inference_mode: bool = False):
        
        # Input validation
        if not sequence_indices_to_use:
            raise ValueError("sequence_indices_to_use cannot be empty")
        if all_ohlcv_data.empty:
            raise ValueError("all_ohlcv_data cannot be empty")
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if time_window <= 0:
            raise ValueError("time_window must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if scaler_std == 0:
            raise ValueError("scaler_std cannot be zero (would cause division by zero)")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if target_batch_time <= 0:
            raise ValueError("target_batch_time must be positive")
        if not (0.0 < jax_memory_fraction <= 1.0):
            raise ValueError("jax_memory_fraction must be between 0.0 and 1.0")
        if jax_recompilation_threshold <= 0:
            raise ValueError("jax_recompilation_threshold must be positive")
        if jax_compile_warmup_batches < 0:
            raise ValueError("jax_compile_warmup_batches must be non-negative")
        
        # Copy the list to avoid in-place shuffling of shared lists
        self.sequence_indices_to_use = list(sequence_indices_to_use)
        self.all_ohlcv_data = all_ohlcv_data
        self.seq_length = seq_length
        self.time_window = time_window
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.batch_size = batch_size
        self.original_batch_size = batch_size  # Keep original for reference
        self.shuffle_indices = shuffle_indices
        self.tickers = tickers
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.adaptive_batch_size = adaptive_batch_size
        self.target_batch_time = target_batch_time
        self.inference_mode = inference_mode

        # JAX-aware adaptive batch size parameters
        self.jax_memory_fraction = jax_memory_fraction
        self.jax_recompilation_threshold = jax_recompilation_threshold
        self.jax_compile_warmup_batches = jax_compile_warmup_batches
        self.jax_max_batch_size = jax_max_batch_size
        self.jax_min_batch_size = jax_min_batch_size
        self.jax_memory_monitoring = jax_memory_monitoring

        # Enhanced features parameters
        self.include_news = include_news
        self.include_text = include_text
        self.news_window = news_window
        self.text_model = text_model
        self.news_cache_dir = news_cache_dir
        
        # Store ticker count for type safety (moved earlier)
        self.num_tickers = len(self.tickers) if self.tickers else 0
        
        # Initialize enhanced features processors if enabled
        self.news_processor = None
        self.text_processor = None
        
        
        
        # Pre-fetch and cache news data once during initialization
        self.cached_news_data = None
        self.cached_news_features = {}
        self.cached_text_features = {}
        
        # Setup date range for news alignment (moved much earlier)
        self._setup_date_range()

        if self.include_news or self.include_text:
            try:
                from src.data.unified_news_data import get_news_for_tickers
                self.news_processor = {
                    'get_news': get_news_for_tickers
                }
                print(f"✅ News processor initialized for {self.num_tickers} tickers")
                
                # Calculate optimal news fetch period based on ticker data range
                self._setup_optimal_news_fetching()
                
            except ImportError as e:
                print(f"⚠️  Warning: Could not import news processor: {e}")
                self.include_news = False
        
        if self.include_text:
            try:
                # Initialize text processor (will be implemented)
                self.text_processor = self._initialize_text_processor()
                print(f"✅ Text processor initialized with model: {self.text_model}")
                
                # Pre-compute text embeddings if news data is available
                if self.cached_news_data is not None and not self.cached_news_data.empty:
                    print("🔄 Pre-computing text embeddings...")
                    self._precompute_text_features()
                    print("✅ Text embeddings pre-computed")
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize text processor: {e}")
                self.include_text = False

        # Performance monitoring
        self.stats = {
            'batches_generated': 0,
            'sequences_processed': 0,
            'sequences_skipped': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_generation_time': 0.0,
            'avg_batch_time': 0.0,
            'batch_size_adjustments': 0,
            'min_batch_size': batch_size,
            'max_batch_size': batch_size,
            # New JAX-aware stats
            'jax_compilations_detected': 0,
            'jax_memory_pressure_events': 0,
            'jax_optimal_batch_size': batch_size,
            'jax_compile_warmup_complete': False
        }
        
        # Adaptive batch size tracking
        if self.adaptive_batch_size:
            self.batch_times = []
            self.adjustment_threshold = 5  # Adjust every N batches
            self.jax_compile_detection_window = []
            self.jax_memory_usage_history = []
            self.jax_optimal_batch_size_history = []
        
        # Caching
        if self.enable_caching:
            self.cache = {}
            self.cache_lock = threading.Lock()

        # Validate tickers list
        if self.tickers is None:
            raise ValueError("tickers list cannot be None")
        
        # Check if all required tickers are present in the data
        if hasattr(self.all_ohlcv_data.columns, 'levels'):
            available_tickers = set(self.all_ohlcv_data.columns.levels[0])
        else:
            available_tickers = set(self.all_ohlcv_data.columns)
        missing_tickers = set(self.tickers) - available_tickers
        if missing_tickers:
            raise ValueError(f"Missing tickers in data: {missing_tickers}")

        # Process and cache data using simplified approach
        self.lookup_array = self.process_and_cache_data(
            all_ohlcv_data=all_ohlcv_data,
            tickers=self.tickers,
            time_window=time_window,
            threshold_percent=0.02,  # Default threshold
            force_recompute=False
        )        

        # Validate sequence indices
        self._validate_sequence_indices()

        if self.shuffle_indices:
            random.shuffle(self.sequence_indices_to_use)

        self.current_idx_in_generator = 0
        self.news_features_array = None
        self.text_features_array = None
        self.date_to_idx = {date: i for i, date in enumerate(self.data_dates)}
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}
        # Precompute news features array if enabled
        if self.include_news and hasattr(self, 'precomputed_news_features') and self.precomputed_news_features:
            num_dates = len(self.data_dates)
            num_tickers = len(self.tickers)
            self.news_features_array = np.zeros((num_dates, num_tickers, 10), dtype=np.float32)
            for (ticker, date), features in self.precomputed_news_features.items():
                if ticker in self.ticker_to_idx and date in self.date_to_idx:
                    t_idx = self.ticker_to_idx[ticker]
                    d_idx = self.date_to_idx[date]
                    self.news_features_array[d_idx, t_idx, :] = features
        # Precompute text features array if enabled
        if self.include_text and self.cached_text_features:
            num_dates = len(self.data_dates)
            num_tickers = len(self.tickers)
            self.text_features_array = np.zeros((num_dates, num_tickers, 384), dtype=np.float32)
            for (ticker, date), features in self.cached_text_features.items():
                if ticker in self.ticker_to_idx and date in self.date_to_idx:
                    t_idx = self.ticker_to_idx[ticker]
                    d_idx = self.date_to_idx[date]
                    self.text_features_array[d_idx, t_idx, :] = features

    def _setup_date_range(self):
        """Setup date range for news alignment."""
        try:
            print(f"🔄 Setting up date range for {self.num_tickers} tickers...")
            
            # Get all close prices for sequence generation and convert to numpy array
            all_close_prices_ordered_df = self.all_ohlcv_data.xs('Close', level=1, axis=1)[self.tickers]
            
            # Store date range for news alignment
            self.data_dates = all_close_prices_ordered_df.index
            self.data_start_date = self.data_dates[0]
            self.data_end_date = self.data_dates[-1]
            
            # Store only numpy array for sequence generation
            self.all_close_prices_np = all_close_prices_ordered_df.values.astype(np.float32)
            # self.all_close_prices_jax = jnp.array(self.all_close_prices_np)  # REMOVE persistent JAX array
            
            print(f"✅ Date range setup complete: {self.data_start_date.date()} to {self.data_end_date.date()}")
            
        except Exception as e:
            print(f"❌ Error in _setup_date_range: {e}")
            raise

    def process_and_cache_data(
        self,
        all_ohlcv_data: pd.DataFrame,
        tickers: List[str],
        time_window: int,
        threshold_percent: float = 0.02,
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Process stock data and cache the results for each time window using JAX GPU acceleration.
        For each ticker and each time point, compute:
        - timestamp
        - ticker
        - close_price
        - return (future return based on time_window)
        - label (action classification)
        
        Args:
            all_ohlcv_data: OHLCV data DataFrame
            tickers: List of ticker symbols
            time_window: Time window for future prices
            threshold_percent: Threshold for action classification
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            Returns a 3D NumPy array: (max_time_idx+1, num_tickers, 3) with [close_price, return, label].
        """
        cache_file = _get_cache_filename(tickers, time_window, threshold_percent)
        
        # Check if cache exists and load if available
        if not force_recompute and cache_file.exists():
            print(f"Loading processed data from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    lookup_array = pickle.load(f)
                print(f"✅ Loaded processed data for {lookup_array.shape[0] * lookup_array.shape[1]} time-ticker points")
                return lookup_array
            except Exception as e:
                print(f"⚠️  Error loading cache: {e}. Recomputing...")
        
        print(f"🔄 Processing data for {len(tickers)} tickers with time window {time_window} using JAX GPU acceleration...")
               
        # Convert to JAX array and transfer to GPU
        all_close_prices_jax = jnp.array(self.all_close_prices_np)
        max_time_idx = all_close_prices_jax.shape[0] - 1
        num_tickers = len(tickers)
        # [close_price, return, label]
        lookup_array = np.full((max_time_idx + 1, num_tickers, 3), np.nan, dtype=np.float32)
        # Define process_ticker_data here
        # JIT-compiled function for processing a single ticker's data
        @partial(jax.jit, static_argnums=(1, 2, 3))
        def process_ticker_data(prices: jnp.ndarray, ticker_idx: int, time_window: int, threshold_percent: float):
            """
            Process all time points for a single ticker using JAX GPU acceleration.
            
            Args:
                prices: JAX array of prices for all tickers
                ticker_idx: Index of the ticker to process
                time_window: Time window for future prices
                threshold_percent: Threshold for action classification
                
            Returns:
                Tuple of (valid_indices, labels, returns, close_prices)
            """
            num_time_points = prices.shape[0] - time_window
            ticker_prices = prices[:, ticker_idx]
            
            # Create arrays for results
            valid_indices = jnp.arange(num_time_points)
            labels = jnp.full(num_time_points, -1, dtype=jnp.int32)  # -1 for invalid
            returns = jnp.zeros(num_time_points, dtype=jnp.float32)
            close_prices = jnp.zeros(num_time_points, dtype=jnp.float32)
            
            def process_time_point(i):
                current_price = ticker_prices[i]
                future_prices = jax.lax.dynamic_slice(ticker_prices, (i + 1,), (time_window,))
                
                # Check validity
                is_valid = (
                    ~jnp.isnan(current_price) & ~jnp.isinf(current_price) & (current_price > 0) &
                    ~jnp.any(jnp.isnan(future_prices)) & ~jnp.any(jnp.isinf(future_prices)) & 
                    jnp.all(future_prices > 0)
                )
                
                # Compute max and min future prices
                max_future_price = jnp.max(future_prices)
                min_future_price = jnp.min(future_prices)
                
                # Determine label and return
                above_threshold = max_future_price > current_price * (1 + threshold_percent)
                below_threshold = min_future_price < current_price * (1 - threshold_percent)
                
                def buy_call():
                    return ACTION_BUY_CALL, (max_future_price - current_price) / current_price
                
                def buy_put():
                    return ACTION_BUY_PUT, (current_price - min_future_price) / current_price
                
                def hold():
                    final_price = future_prices[-1]
                    label = jax.lax.cond(
                        final_price < current_price * (1 - threshold_percent),
                        lambda _: ACTION_BUY_PUT,
                        lambda _: ACTION_HOLD,
                        operand=None
                    )
                    return label, jnp.abs(final_price - current_price) / current_price
                
                label, return_val = jax.lax.cond(
                    above_threshold,
                    lambda _: buy_call(),
                    lambda _: jax.lax.cond(
                        below_threshold,
                        lambda _: buy_put(),
                        lambda _: hold(),
                        operand=None
                    ),
                    operand=None
                )
                
                # Return results
                return (
                    jax.lax.cond(is_valid, lambda _: i, lambda _: -1, operand=None),
                    jax.lax.cond(is_valid, lambda _: label, lambda _: -1, operand=None),
                    jax.lax.cond(is_valid, lambda _: return_val, lambda _: 0.0, operand=None),
                    jax.lax.cond(is_valid, lambda _: current_price, lambda _: 0.0, operand=None)
                )
            
            # Process all time points in parallel
            results = jax.vmap(process_time_point)(jnp.arange(num_time_points))
            valid_indices, labels, returns, close_prices = results
            
            # Return all results and mask for filtering outside JIT context
            valid_mask = valid_indices >= 0
            return valid_indices, labels, returns, close_prices, valid_mask
        
        # Process all tickers using GPU acceleration

        for ticker_idx, ticker in enumerate(tickers):
            print(f"  Processing {ticker} with GPU acceleration...")
            # Process ticker data on GPU
            valid_indices, labels, returns, close_prices, valid_mask = process_ticker_data(
                all_close_prices_jax, ticker_idx, time_window, threshold_percent
            )
            # Filter results based on valid_mask
            filtered_valid_indices = valid_indices[valid_mask]
            filtered_labels = labels[valid_mask]
            filtered_returns = returns[valid_mask]
            filtered_close_prices = close_prices[valid_mask]
            # Convert JAX arrays to Python lists for storage
            valid_indices_np = np.array(filtered_valid_indices)
            labels_np = np.array(filtered_labels)
            returns_np = np.array(filtered_returns)
            close_prices_np = np.array(filtered_close_prices)
            lookup_array[valid_indices_np, ticker_idx, 0] = close_prices_np
            lookup_array[valid_indices_np, ticker_idx, 1] = returns_np
            lookup_array[valid_indices_np, ticker_idx, 2] = labels_np
        # Use -1 for missing labels
        missing = np.isnan(lookup_array[..., 2])
        lookup_array[..., 2][missing] = -1
        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(lookup_array, f)
        print(f"✅ Processed {np.count_nonzero(~np.isnan(lookup_array[...,0]))} time-ticker points using GPU acceleration and saved to cache: {cache_file}")
        return lookup_array


    def _setup_optimal_news_fetching(self):
        """Setup optimal news fetching based on ticker data period."""
        import hashlib, pickle
        try:
            from datetime import datetime, timedelta
            print(f"🔄 Starting news fetching setup...")
            print(f"   Has data_end_date: {hasattr(self, 'data_end_date')}")
            print(f"   Has data_start_date: {hasattr(self, 'data_start_date')}")
            data_period_days = (self.data_end_date - self.data_start_date).days
            total_days_to_fetch = data_period_days + self.news_window + 30  # 30 days buffer
            print(f"🔄 Fetching news data for {self.num_tickers} tickers over {total_days_to_fetch} days...")
            print(f"   Data period: {self.data_start_date.date()} to {self.data_end_date.date()}")
            self.cached_news_data = self.news_processor['get_news'](
                tickers=self.tickers,
                days_back=total_days_to_fetch,
                time_from=self.data_start_date.strftime('%Y%m%dT0000'),
                time_to=self.data_end_date.strftime('%Y%m%dT2359'),
                limit_per_ticker=2 * total_days_to_fetch  # 2x days for better coverage
            )
            if not self.cached_news_data.empty:
                print(f"✅ Fetched {len(self.cached_news_data)} news articles")
                self.cached_news_data['date'] = pd.to_datetime(self.cached_news_data['time_published']).dt.date
                # --- News features cache logic ---
                params_string = f"{self.tickers}_{self.data_start_date}_{self.data_end_date}_{self.news_window}"
                cache_key = f"newsfeat_{hashlib.sha256(params_string.encode()).hexdigest()}.pkl"
                cache_path = DATA_CACHE_DIR / cache_key
                if cache_path.exists():
                    print(f"Loading precomputed news features from cache: {cache_path}")
                    with open(cache_path, 'rb') as f:
                        self.precomputed_news_features = pickle.load(f)
                else:
                    self.precomputed_news_features = {}
                    for (ticker, date), group in self.cached_news_data.groupby(['ticker', 'date']):
                        sentiment_score = group['ticker_sentiment_score'].mean()
                        relevance_score = group['relevance_score'].mean()
                        sentiment_label = self._encode_sentiment_label(
                            group['ticker_sentiment_label'].mode().iloc[0] if not group['ticker_sentiment_label'].empty else 'neutral')
                        overall_sentiment = group['overall_sentiment_score'].mean()
                        overall_label = self._encode_sentiment_label(
                            group['overall_sentiment_label'].mode().iloc[0] if not group['overall_sentiment_label'].empty else 'neutral')
                        news_volume = len(group)
                        source_credibility = self._calculate_source_credibility(group)
                        topic_features = self._extract_topic_features(group)
                        features = np.array([
                            sentiment_score, relevance_score, sentiment_label,
                            overall_sentiment, overall_label, news_volume,
                            source_credibility, topic_features[0], topic_features[1], topic_features[2]
                        ], dtype=np.float32)
                        self.precomputed_news_features[(ticker, date)] = features
                    print(f"✅ Precomputed news features for {len(self.precomputed_news_features)} (ticker, date) pairs")
                    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.precomputed_news_features, f)
                self.news_by_date_ticker = {}
                for ticker in self.tickers:
                    ticker_news = self.cached_news_data[self.cached_news_data['ticker'] == ticker]
                    if not ticker_news.empty:
                        self.news_by_date_ticker[ticker] = ticker_news.groupby('date')
                    else:
                        self.news_by_date_ticker[ticker] = {}
                print(f"✅ Indexed news data for efficient sequence-based slicing")
            else:
                print("⚠️  No news data fetched - continuing without news features")
        except Exception as e:
            print(f"⚠️  Error setting up news fetching: {e}")
            self.cached_news_data = pd.DataFrame()

    def _validate_sequence_indices(self):
        """
        Validate that all sequence indices are within valid bounds.
        For training/backtesting, all indices must be <= len(data) - seq_length - time_window.
        For inference (inference_mode=True), allow the last index (len(data) - seq_length),
        which does not have a label/return but is valid for prediction.
        """
        max_valid_idx = len(self.all_close_prices_np) - self.seq_length - (0 if self.inference_mode else self.time_window)
        if self.sequence_indices_to_use:
            min_idx, max_idx = min(self.sequence_indices_to_use), max(self.sequence_indices_to_use)
            if min_idx < 0 or max_idx > max_valid_idx:
                if self.inference_mode and max_idx == len(self.all_close_prices_np) - self.seq_length:
                    # Allow last index for inference
                    return
                raise ValueError(f"Sequence indices are out of bounds. Min: {min_idx}, Max: {max_idx}, Max valid: {max_valid_idx}")

    def _precompute_text_features(self):
        import hashlib, pickle
        if self.cached_news_data is None or self.cached_news_data.empty:
            return
        try:
            text_tuples = []
            for (ticker, date), group in self.cached_news_data.groupby(['ticker', 'date']):
                titles = group['title'].astype(str).tolist()
                joined_text = ' '.join(titles)
                text_tuples.append((ticker, date, joined_text))
            all_texts = [t[2] for t in text_tuples]
            params_string = f"{self.tickers}_{self.data_start_date}_{self.data_end_date}_{self.news_window}_{self.text_model}"
            cache_key = f"textfeat_{hashlib.sha256(params_string.encode()).hexdigest()}.pkl"
            cache_path = DATA_CACHE_DIR / cache_key
            if cache_path.exists():
                print(f"Loading precomputed text features from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    self.cached_text_features = pickle.load(f)
            else:
                if self.text_processor is not None and all_texts:
                    embeddings = self.text_processor.encode(all_texts, show_progress_bar=True, batch_size=32)
                    self.cached_text_features = {}
                    for i, (ticker, date, _) in enumerate(text_tuples):
                        self.cached_text_features[(ticker, date)] = embeddings[i]
                    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.cached_text_features, f)
            print(f"✅ Precomputed text features for {len(self.cached_text_features)} (ticker, date) pairs")
        except Exception as e:
            print(f"⚠️  Warning: Error pre-computing text features: {e}")

    def _initialize_text_processor(self):
        """Initialize text processor for feature extraction using sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.text_model,device='cpu')
        return model

    def _encode_sentiment_label(self, label: str) -> float:
        """Encode sentiment label to numeric value."""
        label_mapping = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0,
            'Somewhat-Bullish': 0.5,
            'Bullish': 1.0,
            'Somewhat-Bearish': -0.5,
            'Bearish': -1.0
        }
        return label_mapping.get(label, 0.0)

    def _calculate_source_credibility(self, news_data: pd.DataFrame) -> float:
        """Calculate source credibility score."""
        if news_data.empty:
            return 0.0
        
        # Simple credibility scoring based on source
        credible_sources = ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance']
        sources = news_data['source'].value_counts()
        
        credibility_score = 0.0
        total_articles = len(news_data)
        
        for source, count in sources.items():
            if source in credible_sources:
                credibility_score += count * 1.0
            else:
                credibility_score += count * 0.5
        
        return credibility_score / total_articles if total_articles > 0 else 0.0

    def _extract_topic_features(self, news_data: pd.DataFrame) -> List[float]:
        """Extract topic features from news data."""
        if news_data.empty:
            return [0.0, 0.0, 0.0]
        
        # Simple topic detection based on keywords
        earnings_keywords = ['earnings', 'quarterly', 'revenue', 'profit', 'EPS']
        ipo_keywords = ['IPO', 'initial public offering', 'listing', 'debut']
        merger_keywords = ['merger', 'acquisition', 'takeover', 'buyout']
        
        titles = news_data['title'].str.lower()
        
        earnings_score = sum(1 for title in titles if any(keyword in title for keyword in earnings_keywords))
        ipo_score = sum(1 for title in titles if any(keyword in title for keyword in ipo_keywords))
        merger_score = sum(1 for title in titles if any(keyword in title for keyword in merger_keywords))
        
        total_articles = len(news_data)
        
        return [
            earnings_score / total_articles if total_articles > 0 else 0.0,
            ipo_score / total_articles if total_articles > 0 else 0.0,
            merger_score / total_articles if total_articles > 0 else 0.0
        ]

    def __len__(self):
        """Returns the number of batches in the generator."""
        return (len(self.sequence_indices_to_use) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Resets the generator for a new epoch."""
        self.current_idx_in_generator = 0
        if self.shuffle_indices:
            random.shuffle(self.sequence_indices_to_use)
        return self

    def __next__(self):
        """
        Yields the next batch of (X, Y, R) data using a fully vectorized approach.
        """
        if self.current_idx_in_generator >= len(self.sequence_indices_to_use):
            raise StopIteration
        batch_start_time = time.time()

        # Get the next batch of indices
        batch_indices = self.sequence_indices_to_use[
            self.current_idx_in_generator : self.current_idx_in_generator + self.batch_size
        ]
        self.current_idx_in_generator += len(batch_indices)
        if not batch_indices:
            raise StopIteration

        n_samples = len(batch_indices)
        batch_indices_np = np.array(batch_indices)

        # --- Vectorized Batch Generation ---
        # 1. Create a (B, S) matrix of indices using broadcasting
        sequence_range = np.arange(self.seq_length)
        all_indices = batch_indices_np[:, None] + sequence_range  # Shape: (n_samples, seq_length)

        # 2. Extract data for all sequences in the batch at once
        batch_prices = self.all_close_prices_np[all_indices]  # Shape: (n_samples, seq_length, num_tickers)

        # 3. Enhance features for the entire batch
        normalized_prices = (batch_prices - self.scaler_mean) / self.scaler_std
        enhanced_sequence = normalized_prices[..., np.newaxis]  # Shape: (n_samples, seq_length, num_tickers, 1)

        if self.include_news and self.news_features_array is not None:
            batch_news_features = self.news_features_array[all_indices]
            enhanced_sequence = np.concatenate([enhanced_sequence, batch_news_features], axis=-1)

        if self.include_text and self.text_features_array is not None:
            batch_text_features = self.text_features_array[all_indices]
            enhanced_sequence = np.concatenate([enhanced_sequence, batch_text_features], axis=-1)
        
        # 4. Reshape to final model input shape
        seq_len, num_tickers, num_features_per_ticker = enhanced_sequence.shape[1], enhanced_sequence.shape[2], enhanced_sequence.shape[3]
        batch_x_unpadded = enhanced_sequence.reshape(n_samples, seq_len, num_tickers * num_features_per_ticker)

        # --- Vectorized Label/Return Extraction (already efficient) ---
        end_indices = batch_indices_np + self.seq_length - 1
        arr = self.lookup_array[end_indices, :, :]  # shape: (n_samples, num_tickers, 3)
        batch_y_unpadded = np.where(~np.isnan(arr[:, :, 2]), arr[:, :, 2], -1).astype(np.int32)
        batch_r_unpadded = np.where(~np.isnan(arr[:, :, 1]), arr[:, :, 1], 0.0).astype(np.float32)

        # --- Efficient Padding ---
        # Pre-allocate arrays for the full batch size
        final_batch_size = self.batch_size
        num_features = batch_x_unpadded.shape[2]

        batch_x = np.zeros((final_batch_size, self.seq_length, num_features), dtype=np.float32)
        batch_y = -np.ones((final_batch_size, self.num_tickers), dtype=np.int32)
        batch_r = np.zeros((final_batch_size, self.num_tickers), dtype=np.float32)

        # Fill the allocated arrays with the generated data
        batch_x[:n_samples] = batch_x_unpadded
        batch_y[:n_samples] = batch_y_unpadded
        batch_r[:n_samples] = batch_r_unpadded

        # Create padding mask
        padding_mask = np.zeros(final_batch_size, dtype=bool)
        padding_mask[:n_samples] = True

        result = (batch_x, batch_y, batch_r, padding_mask)
        # Update performance stats
        batch_time = time.time() - batch_start_time
        self.stats['batches_generated'] += 1
        self.stats['sequences_processed'] += batch_x.shape[0]
        self.stats['total_generation_time'] += batch_time
        self.stats['avg_batch_time'] = self.stats['total_generation_time'] / self.stats['batches_generated']
        self._adjust_batch_size(batch_time)
        return result

    def get_batch_info(self):
        """Returns information about the current batch configuration."""
        return {
            'total_sequences': len(self.sequence_indices_to_use),
            'batch_size': self.batch_size,
            'num_batches': (len(self.sequence_indices_to_use) + self.batch_size - 1) // self.batch_size,
            'seq_length': self.seq_length,
            'time_window': self.time_window,
            'num_tickers': len(self.tickers) if self.tickers else 0,
            'shuffle_enabled': self.shuffle_indices,
            'caching_enabled': self.enable_caching,
            'cache_size': self.cache_size if self.enable_caching else 0,
            'processed_data_points': self.lookup_array.shape[0] * self.lookup_array.shape[1]
        }

    def get_performance_stats(self):
        """Returns performance statistics including JAX-aware adaptive batch size stats."""
        stats_copy = self.stats.copy()
        if stats_copy['batches_generated'] > 0:
            stats_copy['sequences_per_second'] = (
                stats_copy['sequences_processed'] / stats_copy['total_generation_time']
                if stats_copy['total_generation_time'] > 0 else 0
            )
            if self.enable_caching:
                total_cache_requests = stats_copy['cache_hits'] + stats_copy['cache_misses']
                stats_copy['cache_hit_rate'] = (
                    stats_copy['cache_hits'] / total_cache_requests 
                    if total_cache_requests > 0 else 0
                )
        
        # Add JAX-aware adaptive batch size statistics
        if self.adaptive_batch_size:
            jax_stats = self.get_jax_adaptive_stats()
            stats_copy.update(jax_stats)
            
            # Add batch size efficiency metrics
            if stats_copy['batches_generated'] > 0:
                stats_copy['batch_size_efficiency'] = (
                    stats_copy['sequences_processed'] / 
                    (stats_copy['batches_generated'] * stats_copy['current_batch_size'])
                )
                stats_copy['batch_size_utilization'] = (
                    stats_copy['current_batch_size'] / stats_copy['max_batch_size']
                    if stats_copy['max_batch_size'] > 0 else 0
                )
        
        return stats_copy

    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'batches_generated': 0,
            'sequences_processed': 0,
            'sequences_skipped': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_generation_time': 0.0,
            'avg_batch_time': 0.0,
            'batch_size_adjustments': 0,
            'min_batch_size': self.original_batch_size,
            'max_batch_size': self.original_batch_size
        }

    def clear_cache(self):
        """Clear the sequence cache."""
        if self.enable_caching:
            with self.cache_lock:
                self.cache.clear()
    
    def clear_news_cache(self):
        """Clear the news and text feature caches."""
        self.cached_news_data = None
        self.cached_news_features.clear()
        self.cached_text_features.clear()
        print("✅ News and text feature caches cleared")

    def _adjust_batch_size(self, batch_time):
        """Dynamically adjust batch size based on performance, JAX recompilation, and memory requirements."""
        if not self.adaptive_batch_size:
            return
            
        self.batch_times.append(batch_time)
        
        # Only adjust every N batches to avoid too frequent changes
        if len(self.batch_times) >= self.adjustment_threshold:
            avg_time = sum(self.batch_times) / len(self.batch_times)
            
            # Detect JAX recompilation (sudden increase in batch time)
            if len(self.batch_times) >= 3:
                recent_avg = sum(self.batch_times[-3:]) / 3
                if recent_avg > avg_time * 2.0:  # Sudden 2x increase suggests recompilation
                    self.stats['jax_compilations_detected'] += 1
                    print(f"JAX recompilation detected (batch time: {recent_avg:.3f}s vs avg: {avg_time:.3f}s)")
                    
                    # During recompilation, be more conservative with batch size changes
                    if recent_avg > self.target_batch_time * 1.5:
                        new_batch_size = max(self.jax_min_batch_size, int(self.batch_size * 0.7))
                    else:
                        new_batch_size = self.batch_size  # Keep current size during recompilation
                else:
                    # Normal performance-based adjustment
                    new_batch_size = self._calculate_optimal_batch_size(avg_time)
            else:
                new_batch_size = self._calculate_optimal_batch_size(avg_time)
            
            # Apply JAX memory constraints
            new_batch_size = self._apply_jax_memory_constraints(new_batch_size)
            
            # Apply batch size limits
            if self.jax_max_batch_size is not None:
                new_batch_size = min(new_batch_size, self.jax_max_batch_size)
            new_batch_size = max(new_batch_size, self.jax_min_batch_size)
            new_batch_size = min(new_batch_size, len(self.sequence_indices_to_use))
            
            # Update batch size if changed
            if new_batch_size != self.batch_size:
                old_batch_size = self.batch_size
                self.batch_size = new_batch_size
                self.stats['batch_size_adjustments'] += 1
                self.stats['min_batch_size'] = min(self.stats['min_batch_size'], new_batch_size)
                self.stats['max_batch_size'] = max(self.stats['max_batch_size'], new_batch_size)
                self.stats['jax_optimal_batch_size'] = new_batch_size
                
                print(f"Batch size adjusted: {old_batch_size} → {new_batch_size} "
                      f"(avg time: {avg_time:.3f}s, target: {self.target_batch_time:.3f}s)")
            
            # Reset tracking
            self.batch_times = []
            self.jax_compile_detection_window = []

    def _calculate_optimal_batch_size(self, avg_time):
        """Calculate optimal batch size using exponential scaling and momentum."""
        if avg_time > self.target_batch_time * 1.5:
            # Exponential decay for slow batches
            scale_factor = jnp.exp(-0.3)  # ~0.74
            new_batch_size = max(self.jax_min_batch_size, int(self.batch_size * scale_factor))
        elif avg_time < self.target_batch_time * 0.5:
            # Exponential growth for fast batches
            scale_factor = jnp.exp(0.2)  # ~1.22
            new_batch_size = min(self.jax_max_batch_size, int(self.batch_size * scale_factor))
        elif avg_time < self.target_batch_time * 0.8:
            # Moderate growth for slightly fast batches
            scale_factor = jnp.exp(0.1)  # ~1.11
            new_batch_size = min(self.jax_max_batch_size, int(self.batch_size * scale_factor))
        elif avg_time > self.target_batch_time * 1.2:
            # Moderate decay for slightly slow batches
            scale_factor = jnp.exp(-0.15)  # ~0.86
            new_batch_size = max(self.jax_min_batch_size, int(self.batch_size * scale_factor))
        else:
            # Within acceptable range, keep current size
            new_batch_size = self.batch_size
        
        # Apply momentum to prevent oscillation
        if hasattr(self, 'previous_batch_size'):
            momentum_factor = 0.7
            new_batch_size = int(momentum_factor * self.previous_batch_size + 
                               (1 - momentum_factor) * new_batch_size)
        
        self.previous_batch_size = new_batch_size
        return new_batch_size

    def _apply_jax_memory_constraints(self, proposed_batch_size):
        """Apply JAX memory constraints to batch size."""
        if not self.jax_memory_monitoring:
            return proposed_batch_size
            
        try:
            # Estimate memory usage for the proposed batch size
            estimated_memory_mb = self._estimate_batch_memory_usage(proposed_batch_size)
            
            # Get available JAX device memory
            available_memory_mb = self._get_jax_available_memory()
            
            # Check if we're using too much memory
            if estimated_memory_mb > available_memory_mb * self.jax_memory_fraction:
                # Reduce batch size to fit within memory constraints
                reduction_factor = (available_memory_mb * self.jax_memory_fraction) / estimated_memory_mb
                new_batch_size = int(proposed_batch_size * reduction_factor)
                new_batch_size = max(self.jax_min_batch_size, new_batch_size)
                
                self.stats['jax_memory_pressure_events'] += 1
                print(f"JAX memory pressure detected. Reducing batch size from {proposed_batch_size} to {new_batch_size}")
                return new_batch_size
                
        except Exception as e:
            print(f"Error applying JAX memory constraints: {e}")
            
        return proposed_batch_size

    def _estimate_batch_memory_usage(self, batch_size):
        """Estimate memory usage for a given batch size in MB."""
        # Rough estimation based on data types and shapes
        # sequence_data: (batch_size, seq_length, num_tickers) * 4 bytes (float32)
        sequence_memory = batch_size * self.seq_length * self.num_tickers * 4
        # labels: (batch_size, num_tickers) * 4 bytes (int32)
        label_memory = batch_size * self.num_tickers * 4
        # returns: (batch_size, num_tickers) * 4 bytes (float32)
        return_memory = batch_size * self.num_tickers * 4
        
        total_memory_bytes = sequence_memory + label_memory + return_memory
        return total_memory_bytes / (1024 * 1024)  # Convert to MB

    def _get_jax_available_memory(self):
        """Get available JAX device memory in MB."""
        try:
            # This is a simplified approach - in practice you might want to use
            # jax.device_get() or other JAX-specific memory management
            return 1024 * 1024 * 1024  # 1GB default
        except Exception:
            return 1024 * 1024 * 1024  # 1GB default

    def get_jax_adaptive_stats(self):
        """Get JAX-specific adaptive statistics."""
        return {
            'jax_compilations_detected': self.stats['jax_compilations_detected'],
            'jax_memory_pressure_events': self.stats['jax_memory_pressure_events'],
            'jax_optimal_batch_size': self.stats['jax_optimal_batch_size'],
            'jax_compile_warmup_complete': self.stats['jax_compile_warmup_complete'],
            'current_batch_size': self.batch_size,
            'memory_fraction': self.jax_memory_fraction
        }

    def __del__(self):
        """Cleanup when generator is destroyed."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'cache') and self.enable_caching:
            self.clear_cache()
        
        # Clear news and text feature caches
        if hasattr(self, 'cached_news_data'):
            self.clear_news_cache()

    def get_memory_usage(self):
        """Get current memory usage statistics."""
        memory_info = {
            'cache_size': len(self.cache) if self.enable_caching else 0,
            'cache_memory_mb': len(self.cache) * 0.1,  # Rough estimate
            'price_data_memory_mb': self.all_close_prices_np.nbytes / (1024 * 1024),
            'total_sequences': self.lookup_array.shape[0] * self.lookup_array.shape[1]
        }
        
        # Add news cache memory usage
        if self.cached_news_data is not None:
            memory_info.update({
                'news_data_memory_mb': len(self.cached_news_data) * 0.01,  # Rough estimate per article
                'text_features_memory_mb': len(self.cached_text_features) * 384 * 4 / (1024 * 1024),  # 384 features * 4 bytes per float32
                'news_features_enabled': self.include_news,
                'text_features_enabled': self.include_text
            })
        
        return memory_info

    def optimize_for_memory(self, max_memory_mb: float = 1000):
        """Optimize memory usage by adjusting cache size and batch size."""
        current_memory = self.get_memory_usage()
        total_memory = current_memory['cache_memory_mb'] + current_memory['price_data_memory_mb']
        
        if total_memory > max_memory_mb:
            # Reduce cache size
            if self.enable_caching:
                target_cache_size = int(self.cache_size * (max_memory_mb / total_memory))
                self.cache_size = max(100, target_cache_size)
                print(f"Reduced cache size to {self.cache_size} for memory optimization")
            
            # Reduce batch size if needed
            if self.batch_size > self.jax_min_batch_size:
                self.batch_size = max(self.jax_min_batch_size, int(self.batch_size * 0.8))
                print(f"Reduced batch size to {self.batch_size} for memory optimization")

    
    def get_news_cache_stats(self):
        """Get detailed statistics about the news cache."""
        if self.cached_news_data is None:
            return {
                'news_cache_enabled': False,
                'articles_cached': 0,
                'tickers_with_news': 0,
                'text_features_cached': 0,
                'cache_hit_rate': 0.0,
                'data_period_aligned': False
            }
        
        stats = {
            'news_cache_enabled': True,
            'articles_cached': len(self.cached_news_data),
            'tickers_with_news': self.cached_news_data['ticker'].nunique() if not self.cached_news_data.empty else 0,
            'text_features_cached': len(self.cached_text_features),
            'news_window_days': self.news_window,
            'include_news_features': self.include_news,
            'include_text_features': self.include_text,
            'data_period_aligned': True,
            'data_start_date': str(self.data_start_date.date()) if hasattr(self, 'data_start_date') else None,
            'data_end_date': str(self.data_end_date.date()) if hasattr(self, 'data_end_date') else None,
            'indexed_by_date': hasattr(self, 'news_by_date_ticker'),
            'dynamic_slicing_enabled': True
        }
        
        # Calculate cache hit rate if we have usage statistics
        if hasattr(self, 'stats') and 'news_cache_hits' in self.stats:
            total_requests = self.stats.get('news_cache_hits', 0) + self.stats.get('news_cache_misses', 0)
            if total_requests > 0:
                stats['cache_hit_rate'] = self.stats.get('news_cache_hits', 0) / total_requests
            else:
                stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def get_sequence_news_info(self, start_idx: int) -> Dict[str, Any]:
        """
        Get news information for a specific sequence.
        
        Args:
            start_idx: Starting index for the sequence
            
        Returns:
            Dictionary with news information for the sequence period
        """
        if not self.include_news or self.cached_news_data is None or self.cached_news_data.empty:
            return {
                'sequence_start_date': str(self.data_dates[start_idx].date()) if start_idx < len(self.data_dates) else None,
                'sequence_end_date': str(self.data_dates[min(start_idx + self.seq_length - 1, len(self.data_dates) - 1)].date()) if start_idx < len(self.data_dates) else None,
                'news_available': False
            }
        
        try:
            start_date = self.data_dates[start_idx]
            end_date = self.data_dates[min(start_idx + self.seq_length - 1, len(self.data_dates) - 1)]
            
            sequence_news_info = {
                'sequence_start_date': str(start_date.date()),
                'sequence_end_date': str(end_date.date()),
                'news_available': True,
                'tickers_with_news': {},
                'total_articles_in_period': 0
            }
            
            # Check news availability for each ticker in the sequence period
            for ticker in self.tickers:
                ticker_news_groups = self.news_by_date_ticker.get(ticker, {})
                articles_in_period = 0
                dates_with_news = []
                
                for seq_idx in range(self.seq_length):
                    current_date = self.data_dates[start_idx + seq_idx]
                    date_key = current_date.date()
                    
                    if date_key in ticker_news_groups:
                        day_news = ticker_news_groups[date_key]
                        if not day_news.empty:
                            articles_in_period += len(day_news)
                            dates_with_news.append(str(date_key))
                
                sequence_news_info['tickers_with_news'][ticker] = {
                    'articles_in_period': articles_in_period,
                    'dates_with_news': dates_with_news
                }
                sequence_news_info['total_articles_in_period'] += articles_in_period
            
            return sequence_news_info
            
        except Exception as e:
            print(f"⚠️  Warning: Error getting sequence news info: {e}")
            return {
                'sequence_start_date': str(self.data_dates[start_idx].date()) if start_idx < len(self.data_dates) else None,
                'sequence_end_date': str(self.data_dates[min(start_idx + self.seq_length - 1, len(self.data_dates) - 1)].date()) if start_idx < len(self.data_dates) else None,
                'news_available': False,
                'error': str(e)
            }


class _BatchReshaper:
    """
    Lazily reshapes batches from the base generator to (n_devices, per_device_batch_size, ...),
    splitting only the first (batch) axis and leaving all other axes untouched.
    """
    def __init__(self, base_generator, n_devices):
        self.base_generator = base_generator
        self.n_devices = n_devices
        self._reset()

    def _reset(self):
        self.base_iter = iter(self.base_generator)

    def __iter__(self):
        return self

    def __next__(self):
        batch_x, batch_y, batch_r, padding_mask = next(self.base_iter)
        ldc = self.n_devices
        # Only split the first axis, leave the rest untouched, use -1 for per-device batch size
        batch_x = batch_x.reshape((ldc, -1) + batch_x.shape[1:])
        batch_y = batch_y.reshape((ldc, -1) + batch_y.shape[1:])
        batch_r = batch_r.reshape((ldc, -1) + batch_r.shape[1:])
        padding_mask = padding_mask.reshape((ldc, -1) + padding_mask.shape[1:])
        return (batch_x, batch_y, batch_r, padding_mask)

class PrefetchGenerator:
    """
    Prefetch generator that uses flax.jax_utils.prefetch_to_device to prefetch batches to device in the background.
    This enables true overlap of data transfer and model computation in JAX.
    """
    def __init__(self, generator, buffer_size=5, devices=None):
        self.generator = generator
        self.buffer_size = buffer_size
        self.devices = devices if devices is not None else jax.devices()
        self.n_devices = len(self.devices)
        # Wrap the generator with the lazy reshaper
        self._reshaped_gen = _BatchReshaper(self.generator, self.n_devices)

    def __len__(self):
        return len(self.generator)

    def __iter__(self):
        self._reshaped_gen._reset()
        reshaped_iter = iter(self._reshaped_gen)  # Reset the reshaper's iterator for a new epoch
        self._prefetch_iter = prefetch_to_device(reshaped_iter, size=self.buffer_size, devices=self.devices)
        return self

    def __next__(self):
        """Get the next batch from the prefetch_to_device iterator."""
        return next(self._prefetch_iter)

    def get_status(self):
        """Return a dummy status for compatibility."""
        return {
            'buffer_size': self.buffer_size,
            'prefetch_to_device': True
        }

# --- Utility functions for data generation ---
def get_ticker_to_column_map(df: pd.DataFrame) -> Dict[str, int]:
    """Get a map from ticker to column index."""
    if isinstance(df.columns, pd.MultiIndex):
        # Assuming the top level of the MultiIndex contains tickers
        return {ticker: i for i, ticker in enumerate(df.columns.levels[0])}
    return {col: i for i, col in enumerate(df.columns)} 