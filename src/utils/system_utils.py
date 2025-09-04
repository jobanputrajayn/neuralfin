"""
System utilities for the Hyper framework.

Contains functions for system information, monitoring, and financial calculations.
"""

import psutil
import platform
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import warnings


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        dict: System information including CPU, memory, and platform details
    """
    try:
        cpu_info = {
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True),
            'usage_percent': psutil.cpu_percent(interval=1),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
    except Exception:
        cpu_info = {'error': 'Could not retrieve CPU information'}

    try:
        memory_info = psutil.virtual_memory()._asdict()
        memory_info['total_gb'] = memory_info['total'] / (1024**3)
        memory_info['available_gb'] = memory_info['available'] / (1024**3)
        memory_info['used_gb'] = memory_info['used'] / (1024**3)
    except Exception:
        memory_info = {'error': 'Could not retrieve memory information'}

    try:
        disk_info = psutil.disk_usage('/')._asdict()
        disk_info['total_gb'] = disk_info['total'] / (1024**3)
        disk_info['free_gb'] = disk_info['free'] / (1024**3)
        disk_info['used_gb'] = disk_info['used'] / (1024**3)
    except Exception:
        disk_info = {'error': 'Could not retrieve disk information'}

    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu': cpu_info,
        'memory': memory_info,
        'disk': disk_info
    }


def calculate_financial_metrics(equity_curve: pd.Series, 
                              risk_free_rate: float = 0.02,
                              trading_days_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate comprehensive financial metrics for an equity curve.
    
    Args:
        equity_curve: Pandas Series with equity values over time
        risk_free_rate: Annual risk-free rate (default: 2%)
        trading_days_per_year: Number of trading days per year (default: 252)
        
    Returns:
        dict: Dictionary containing financial metrics
    """
    if len(equity_curve) < 2:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': np.nan,
            'max_drawdown': 0.0,
            'calmar_ratio': np.nan,
            'sortino_ratio': np.nan,
            'var_95': np.nan,
            'cvar_95': np.nan
        }
    
    # Calculate daily returns
    daily_returns = equity_curve.pct_change().dropna()
    
    if len(daily_returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': np.nan,
            'max_drawdown': 0.0,
            'calmar_ratio': np.nan,
            'sortino_ratio': np.nan,
            'var_95': np.nan,
            'cvar_95': np.nan
        }
    
    # Total return (geometric)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # Annualized return (compound annual growth rate)
    trading_days = len(daily_returns)
    annualized_return = ((1 + total_return) ** (trading_days_per_year / trading_days)) - 1
    
    # Annualized volatility
    annualized_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
    
    # Sharpe ratio with risk-free rate
    risk_free_rate_daily = (1 + risk_free_rate) ** (1/trading_days_per_year) - 1
    excess_return_annual = annualized_return - risk_free_rate
    sharpe_ratio = excess_return_annual / annualized_volatility if annualized_volatility > 0 else np.nan
    
    # Maximum drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    # Sortino ratio (using downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(trading_days_per_year) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return_annual / downside_deviation if downside_deviation > 0 else np.nan
    
    # Value at Risk (VaR) and Conditional VaR (CVaR) at 95% confidence
    var_95 = np.percentile(daily_returns, 5)  # 5th percentile (95% VaR)
    cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else np.nan
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95
    }


def validate_numerical_inputs(data: np.ndarray, 
                             name: str = "data",
                             check_nan: bool = True,
                             check_inf: bool = True,
                             check_finite: bool = True) -> bool:
    """
    Validate numerical inputs for common issues.
    
    Args:
        data: Input array to validate
        name: Name of the data for error messages
        check_nan: Whether to check for NaN values
        check_inf: Whether to check for infinite values
        check_finite: Whether to check for finite values
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if check_nan and np.isnan(data).any():
            warnings.warn(f"NaN values detected in {name}")
            return False
            
        if check_inf and np.isinf(data).any():
            warnings.warn(f"Infinite values detected in {name}")
            return False
            
        if check_finite and not np.isfinite(data).all():
            warnings.warn(f"Non-finite values detected in {name}")
            return False
            
        return True
        
    except Exception as e:
        warnings.warn(f"Error validating {name}: {e}")
        return False


def calculate_rolling_metrics(data: pd.Series, 
                            window: int = 20,
                            metrics: list = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Calculate rolling statistics for time series data.
    
    Args:
        data: Pandas Series with time series data
        window: Rolling window size
        metrics: List of metrics to calculate
        
    Returns:
        pd.DataFrame: DataFrame with rolling metrics
    """
    result = pd.DataFrame(index=data.index)
    
    for metric in metrics:
        if metric == 'mean':
            result[f'rolling_{metric}'] = data.rolling(window=window).mean()
        elif metric == 'std':
            result[f'rolling_{metric}'] = data.rolling(window=window).std()
        elif metric == 'min':
            result[f'rolling_{metric}'] = data.rolling(window=window).min()
        elif metric == 'max':
            result[f'rolling_{metric}'] = data.rolling(window=window).max()
        elif metric == 'median':
            result[f'rolling_{metric}'] = data.rolling(window=window).median()
        elif metric == 'skew':
            result[f'rolling_{metric}'] = data.rolling(window=window).skew()
        elif metric == 'kurt':
            result[f'rolling_{metric}'] = data.rolling(window=window).kurt()
    
    return result


def detect_outliers(data: np.ndarray, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in numerical data.
    
    Args:
        data: Input array
        method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        np.ndarray: Boolean array indicating outliers
    """
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for returns data.
    
    Args:
        returns_df: DataFrame with returns data (columns are assets, rows are time periods)
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    # Remove any columns with all NaN values
    returns_df = returns_df.dropna(axis=1, how='all')
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    return corr_matrix


def calculate_portfolio_weights(returns_df: pd.DataFrame, 
                              method: str = 'equal',
                              **kwargs) -> pd.Series:
    """
    Calculate portfolio weights using various methods.
    
    Args:
        returns_df: DataFrame with returns data
        method: Weighting method ('equal', 'market_cap', 'risk_parity', 'min_variance')
        **kwargs: Additional arguments for specific methods
        
    Returns:
        pd.Series: Portfolio weights
    """
    if method == 'equal':
        n_assets = len(returns_df.columns)
        return pd.Series(1.0 / n_assets, index=returns_df.columns)
    
    elif method == 'min_variance':
        # Minimum variance portfolio
        cov_matrix = returns_df.cov()
        inv_cov = np.linalg.inv(cov_matrix.values)
        ones = np.ones(len(cov_matrix))
        weights = inv_cov @ ones
        weights = weights / weights.sum()
        return pd.Series(weights, index=returns_df.columns)
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")


def print_system_status():
    """Print current system status to console."""
    info = get_system_info()
    
    print("=== System Status ===")
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version']}")
    
    if 'error' not in info['cpu']:
        print(f"CPU: {info['cpu']['count']} cores, {info['cpu']['usage_percent']:.1f}% usage")
    
    if 'error' not in info['memory']:
        print(f"Memory: {info['memory']['used_gb']:.1f}GB / {info['memory']['total_gb']:.1f}GB "
              f"({info['memory']['percent']:.1f}%)")
    
    if 'error' not in info['disk']:
        print(f"Disk: {info['disk']['used_gb']:.1f}GB / {info['disk']['total_gb']:.1f}GB "
              f"({info['disk']['percent']:.1f}%)")


def get_memory_usage() -> Dict[str, float]:
    """
    Gets current memory usage information.
    
    Returns:
        dict: Dictionary containing memory usage in GB
    """
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'free_gb': memory.free / (1024**3),
        'percent_used': memory.percent
    }


def get_cpu_usage() -> Dict[str, float]:
    """
    Gets current CPU usage information.
    
    Returns:
        dict: Dictionary containing CPU usage information
    """
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'cpu_percent_per_core': psutil.cpu_percent(interval=1, percpu=True),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    }


def get_disk_usage(path: str = "/") -> Dict[str, float]:
    """
    Gets disk usage for a specific path.
    
    Args:
        path (str): Path to check disk usage for
        
    Returns:
        dict: Dictionary containing disk usage in GB
    """
    try:
        usage = psutil.disk_usage(path)
        return {
            'total_gb': usage.total / (1024**3),
            'used_gb': usage.used / (1024**3),
            'free_gb': usage.free / (1024**3),
            'percent_used': usage.percent
        }
    except (PermissionError, FileNotFoundError):
        return {} 