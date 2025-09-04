"""
GPU utilities for the Hyper framework.

Contains functions for checking GPU availability, utilization, and optimization.
"""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import threading
import queue

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
except Exception:
    NVML_AVAILABLE = False


def get_gpu_info():
    """
    Get GPU information in a simplified format.
    
    Returns:
        dict: Dictionary containing GPU information
    """
    gpu_info = check_gpu_availability()
    return {
        'nvml_available': gpu_info['nvml_available'],
        'gpu_count': gpu_info['gpu_count']
    }


def check_gpu_availability():
    """
    Checks GPU availability and configuration.
    
    Returns:
        dict: Dictionary containing GPU information and status
    """
    gpu_info = {
        'nvml_available': NVML_AVAILABLE,
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory': [],
        'gpu_utilization': []
    }
    
    if NVML_AVAILABLE:
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_info['gpu_count'] = gpu_count
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Handle GPU name (could be bytes or string)
                if isinstance(name, bytes):
                    gpu_name = name.decode('utf-8')
                else:
                    gpu_name = str(name)
                
                gpu_info['gpu_names'].append(gpu_name)
                gpu_info['gpu_memory'].append({
                    'total': float(memory_info.total),
                    'free': float(memory_info.free),
                    'used': float(memory_info.used)
                })
                gpu_info['gpu_utilization'].append({
                    'gpu': utilization.gpu,
                    'memory': utilization.memory
                })
        except Exception as e:
            print(f"Error getting GPU information: {e}")
    
    return gpu_info


def get_gpu_utilization(device_index=0):
    """
    Gets GPU utilization for a specific device.
    
    Args:
        device_index (int): GPU device index
        
    Returns:
        float: GPU utilization percentage, or 0.0 if not available
    """
    if not NVML_AVAILABLE:
        return 0.0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    except Exception:
        return 0.0


def get_gpu_memory_info(device_index=0):
    """
    Gets GPU memory information for a specific device.
    
    Args:
        device_index (int): GPU device index
        
    Returns:
        dict: Memory information or None if not available
    """
    if not NVML_AVAILABLE:
        return None
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'total': memory_info.total,
            'free': memory_info.free,
            'used': memory_info.used,
            'utilization_percent': (memory_info.used / memory_info.total) * 100
        }
    except Exception:
        return None


def print_gpu_status():
    """
    Prints current GPU status to console.
    """
    gpu_info = check_gpu_availability()
    
    print("=== GPU Status ===")
    print(f"NVML Available: {gpu_info['nvml_available']}")
    print(f"GPU Count: {gpu_info['gpu_count']}")
    
    if gpu_info['gpu_count'] > 0 and len(gpu_info['gpu_names']) > 0:
        for i in range(gpu_info['gpu_count']):
            if i < len(gpu_info['gpu_names']):
                print(f"\nGPU {i}: {gpu_info['gpu_names'][i]}")
                
                if i < len(gpu_info['gpu_memory']):
                    memory = gpu_info['gpu_memory'][i]
                    # Handle potential string/bytes values
                    total_memory = memory.get('total', 0)
                    used_memory = memory.get('used', 0)
                    free_memory = memory.get('free', 0)
                    
                    # Convert to float if needed
                    if isinstance(total_memory, (str, bytes)):
                        total_memory = float(total_memory)
                    if isinstance(used_memory, (str, bytes)):
                        used_memory = float(used_memory)
                    if isinstance(free_memory, (str, bytes)):
                        free_memory = float(free_memory)
                    
                    memory_gb = total_memory / (1024**3)
                    memory_used_gb = used_memory / (1024**3)
                    memory_free_gb = free_memory / (1024**3)
                    
                    print(f"  Memory: {memory_used_gb:.1f}GB / {memory_gb:.1f}GB (Free: {memory_free_gb:.1f}GB)")
                
                if i < len(gpu_info['gpu_utilization']):
                    util = gpu_info['gpu_utilization'][i]
                    print(f"  Utilization: GPU {util['gpu']}%, Memory {util['memory']}%")
    else:
        print("No GPUs detected or NVML not available")


class JAXGPUManager:
    """
    Manages JAX GPU operations to minimize data transfers and maximize GPU utilization.
    """
    
    def __init__(self, device_index: int = 0, memory_fraction: float = 0.8):
        """
        Initialize GPU manager.
        
        Args:
            device_index: GPU device to use
            memory_fraction: Fraction of GPU memory to use
        """
        self.device_index = device_index
        self.memory_fraction = memory_fraction
        self.device = jax.devices()[device_index] if jax.device_count() > 0 else None
        
        # Data transfer tracking
        self.transfer_stats = {
            'cpu_to_gpu_transfers': 0,
            'gpu_to_cpu_transfers': 0,
            'total_transfer_time': 0.0,
            'total_compute_time': 0.0,
            'data_kept_on_gpu': 0
        }
        
        # GPU memory tracking
        self.gpu_memory_usage = 0
        self.max_gpu_memory_usage = 0
        
        # Pre-allocated buffers on GPU
        self.gpu_buffers = {}
        self.buffer_lock = threading.Lock()
        
        print(f"JAX GPU Manager initialized on device: {self.device}")
    
    def optimize_jax_config(self):
        """Configure JAX for optimal GPU performance."""
        # Set memory fraction
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.memory_fraction)
        
        # Enable XLA optimizations
        jax.config.update('jax_enable_x64', False)  # Use float32 for memory efficiency
        jax.config.update('jax_platform_name', 'gpu')
        
        # Enable aggressive optimizations
        jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
        jax.config.update('jax_debug_nans', False)
        jax.config.update('jax_debug_infs', False)
        
        print("JAX configuration optimized for GPU performance")
    
    def to_device(self, data: np.ndarray, name: Optional[str] = None) -> jnp.ndarray:
        """
        Transfer data to GPU with caching and optimization.
        
        Args:
            data: NumPy array to transfer
            name: Optional name for caching (if None, will use data hash)
            
        Returns:
            JAX array on GPU
        """
        if name is None:
            # Use data hash as name for caching
            name = f"data_{data.shape}_{data.dtype}_{id(data)}"
        
        # Check cache first
        if name in self.gpu_buffers:
            self.transfer_stats['data_kept_on_gpu'] += 1
            return self.gpu_buffers[name]
        
        # Transfer to GPU
        start_time = time.time()
        jax_data = jax.device_put(data, self.device)
        transfer_time = time.time() - start_time
        
        # Update stats
        self.transfer_stats['cpu_to_gpu_transfers'] += 1
        self.transfer_stats['total_transfer_time'] += transfer_time
        
        # Cache the result
        with self.buffer_lock:
            self.gpu_buffers[name] = jax_data
        
        # Manage cache size
        if len(self.gpu_buffers) > 100:
            # Remove oldest entry
            oldest_key = next(iter(self.gpu_buffers))
            with self.buffer_lock:
                del self.gpu_buffers[oldest_key]
        
        # Update memory usage
        data_size = data.nbytes / (1024 * 1024)  # MB
        self.gpu_memory_usage += data_size
        self.max_gpu_memory_usage = max(self.max_gpu_memory_usage, self.gpu_memory_usage)
        
        return jax_data
    
    def from_device(self, jax_data: jnp.ndarray, name: Optional[str] = None) -> np.ndarray:
        """
        Transfer data from GPU to CPU.
        
        Args:
            jax_data: JAX array on GPU
            name: Optional name for tracking
            
        Returns:
            NumPy array on CPU
        """
        if name is None:
            name = f"result_{self.transfer_stats['gpu_to_cpu_transfers']}"
        
        # Transfer from GPU
        start_time = time.time()
        cpu_data = jax.device_get(jax_data)
        transfer_time = time.time() - start_time
        
        # Update stats
        self.transfer_stats['gpu_to_cpu_transfers'] += 1
        self.transfer_stats['total_transfer_time'] += transfer_time
        
        # Update memory usage
        data_size = cpu_data.nbytes / (1024 * 1024)  # MB
        self.gpu_memory_usage = max(0, self.gpu_memory_usage - data_size)
        
        return cpu_data
    
    def keep_on_device(self, jax_data: jnp.ndarray, name: Optional[str] = None):
        """
        Keep data on GPU device for reuse.
        
        Args:
            jax_data: JAX array to keep on GPU
            name: Name for the cached data
        """
        if name is not None:
            with self.buffer_lock:
                self.gpu_buffers[name] = jax_data
            self.transfer_stats['data_kept_on_gpu'] += 1
    
    def clear_buffer(self, name: str):
        """Clear a specific buffer from GPU memory."""
        if name in self.gpu_buffers:
            with self.buffer_lock:
                del self.gpu_buffers[name]
    
    def clear_all_buffers(self):
        """Clear all cached buffers from GPU memory."""
        with self.buffer_lock:
            self.gpu_buffers.clear()
        self.gpu_memory_usage = 0
    
    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get data transfer statistics."""
        stats = self.transfer_stats.copy()
        
        # Calculate efficiency metrics
        total_transfers = stats['cpu_to_gpu_transfers'] + stats['gpu_to_cpu_transfers']
        if total_transfers > 0:
            stats['transfer_efficiency'] = stats['data_kept_on_gpu'] / total_transfers
            stats['avg_transfer_time'] = stats['total_transfer_time'] / total_transfers
        else:
            stats['transfer_efficiency'] = 0.0
            stats['avg_transfer_time'] = 0.0
        
        stats['gpu_memory_usage_mb'] = self.gpu_memory_usage
        stats['max_gpu_memory_usage_mb'] = self.max_gpu_memory_usage
        stats['cached_buffers'] = len(self.gpu_buffers)
        
        return stats
    
    def print_transfer_stats(self):
        """Print data transfer statistics."""
        stats = self.get_transfer_stats()
        
        print("\n=== JAX GPU Transfer Statistics ===")
        print(f"CPU → GPU transfers: {stats['cpu_to_gpu_transfers']}")
        print(f"GPU → CPU transfers: {stats['gpu_to_cpu_transfers']}")
        print(f"Data kept on GPU: {stats['data_kept_on_gpu']}")
        print(f"Transfer efficiency: {stats['transfer_efficiency']:.2%}")
        print(f"Total transfer time: {stats['total_transfer_time']:.3f}s")
        print(f"Average transfer time: {stats['avg_transfer_time']:.3f}s")
        print(f"GPU memory usage: {stats['gpu_memory_usage_mb']:.1f}MB")
        print(f"Max GPU memory usage: {stats['max_gpu_memory_usage_mb']:.1f}MB")
        print(f"Cached buffers: {stats['cached_buffers']}")


class CPUDeviceManager:
    """
    Manages CPU operations, mimicking the JAXGPUManager interface
    for environments without a GPU.
    """

    def __init__(self):
        self.transfer_stats = {
            'cpu_to_gpu_transfers': 0,
            'gpu_to_cpu_transfers': 0,
            'total_transfer_time': 0.0,
            'total_compute_time': 0.0,
            'data_kept_on_gpu': 0
        }
        self.device = jax.devices('cpu')[0] # Explicitly set device to CPU
        print("CPU Device Manager initialized.")

    def optimize_jax_config(self):
        """Configure JAX for CPU performance (no-op for CPU)."""
        jax.config.update('jax_platform_name', 'cpu')
        print("JAX configuration optimized for CPU performance")

    def to_device(self, data: np.ndarray, name: Optional[str] = None) -> jnp.ndarray:
        """
        Returns data as a JAX array on CPU.
        Args:
            data: NumPy array to convert
            name: Optional name (ignored for CPU)
        Returns:
            JAX array on CPU
        """
        return jnp.asarray(data)

    def from_device(self, jax_data: jnp.ndarray, name: Optional[str] = None) -> np.ndarray:
        """
        Returns data as a NumPy array (already on CPU).
        Args:
            jax_data: JAX array on CPU
            name: Optional name (ignored for CPU)
        Returns:
            NumPy array
        """
        return np.asarray(jax_data)

    def keep_on_device(self, jax_data: jnp.ndarray, name: Optional[str] = None):
        """No-op for CPU."""
        pass

    def clear_buffer(self, name: str):
        """No-op for CPU."""
        pass

    def clear_all_buffers(self):
        """No-op for CPU."""
        pass

    def get_transfer_stats(self) -> Dict[str, Any]:
        """Returns empty transfer stats for CPU."""
        return self.transfer_stats

    def print_transfer_stats(self):
        """Prints CPU transfer stats (always zero)."""
        print("CPU Device Manager: No data transfers occurred.")


class GPUBatchOptimizer:
    """
    Optimizes batch size dynamically based on GPU memory and utilization.
    """

    def __init__(self, gpu_manager: Union[JAXGPUManager, CPUDeviceManager], batch_size: int = 64):
        """
        Initialize GPU batch optimizer.

        Args:
            gpu_manager: Instance of JAXGPUManager or CPUDeviceManager
            batch_size: Initial batch size
        """
        self.gpu_manager = gpu_manager
        self.batch_size = batch_size
        self.current_max_batch_size = batch_size  # Initialize with initial batch size
        self.optimizer_stats = {
            'optimized_batch_sizes': [],
            'batch_optimization_history': []
        }
        self.last_sample_time = 0.0
        self.last_gpu_util = 0.0

    def optimize_batch_size(self, sample_time: float, gpu_util: float) -> int:
        """
        Optimizes batch size based on recent sample time and GPU utilization.

        Args:
            sample_time: Time taken for a batch (seconds)
            gpu_util: Current GPU utilization percentage

        Returns:
            int: Recommended batch size
        """
        # For CPU, batch size optimization is not relevant, return current batch size
        if isinstance(self.gpu_manager, CPUDeviceManager):
            return self.batch_size

        # If we have a GPU, proceed with dynamic batch size adjustment
        current_memory_info = get_gpu_memory_info(self.gpu_manager.device_index)

        if current_memory_info and current_memory_info['utilization_percent'] > 85 and self.batch_size > 1:
            # Reduce batch size if memory utilization is too high
            new_batch_size = max(1, self.batch_size // 2)
            print(f"[GPU Optimizer] Reducing batch size due to high memory utilization: {self.batch_size} -> {new_batch_size}")
            self.batch_size = new_batch_size
            self.optimizer_stats['batch_optimization_history'].append({
                'timestamp': time.time(),
                'event': 'memory_reduction',
                'old_batch_size': self.batch_size * 2, # Store the old batch size before reduction
                'new_batch_size': self.batch_size,
                'memory_utilization': current_memory_info['utilization_percent']
            })
            return self.batch_size

        # Dynamic adjustment based on sample time and GPU utilization
        # This part needs careful tuning and might involve more complex heuristics
        if sample_time > 0.0 and gpu_util > 0.0:
            # Example heuristic: if GPU utilization is low, try increasing batch size
            # If utilization is high, and performance is good, maintain or slightly increase
            target_gpu_util = 80  # Aim for 80% utilization
            adjustment_factor = gpu_util / target_gpu_util

            if adjustment_factor < 0.9 and self.batch_size < self.current_max_batch_size:
                new_batch_size = min(self.current_max_batch_size, int(self.batch_size * 1.2))
                if new_batch_size > self.batch_size:
                    print(f"[GPU Optimizer] Increasing batch size due to low utilization: {self.batch_size} -> {new_batch_size}")
                    self.batch_size = new_batch_size
                    self.optimizer_stats['batch_optimization_history'].append({
                        'timestamp': time.time(),
                        'event': 'utilization_increase',
                        'old_batch_size': self.batch_size / 1.2, # Store the old batch size before increase
                        'new_batch_size': self.batch_size,
                        'gpu_utilization': gpu_util
                    })
            elif adjustment_factor > 1.1 and self.batch_size > 1:
                new_batch_size = max(1, int(self.batch_size * 0.8))
                if new_batch_size < self.batch_size:
                    print(f"[GPU Optimizer] Decreasing batch size due to high utilization: {self.batch_size} -> {new_batch_size}")
                    self.batch_size = new_batch_size
                    self.optimizer_stats['batch_optimization_history'].append({
                        'timestamp': time.time(),
                        'event': 'utilization_decrease',
                        'old_batch_size': self.batch_size / 0.8, # Store the old batch size before decrease
                        'new_batch_size': self.batch_size,
                        'gpu_utilization': gpu_util
                    })

        # Update current_max_batch_size if memory allows
        if current_memory_info:
            available_memory_mb = (current_memory_info['total'] - current_memory_info['used']) / (1024 * 1024) # Convert bytes to MB
            self.current_max_batch_size = self._calculate_max_batch_size(available_memory_mb)

        self.optimizer_stats['optimized_batch_sizes'].append(self.batch_size)
        self.last_sample_time = sample_time
        self.last_gpu_util = gpu_util

        return self.batch_size

    def _get_available_gpu_memory(self) -> float:
        """
        Gets available GPU memory in MB.
        
        Returns:
            float: Available memory in MB, or 0.0 if not available
        """
        if isinstance(self.gpu_manager, CPUDeviceManager):
            return 0.0 # No GPU memory to report for CPU

        memory_info = get_gpu_memory_info(self.gpu_manager.device_index)
        if memory_info:
            return memory_info['free'] / (1024 * 1024)  # Convert bytes to MB
        return 0.0

    def _calculate_max_batch_size(self, available_memory_mb: float) -> int:
        """
        Estimates maximum possible batch size based on available GPU memory.
        
        This is a very rough estimation and needs to be refined based on model size
        and actual memory usage patterns. Assuming each item in a batch takes a
        certain amount of memory.
        
        Args:
            available_memory_mb: Available GPU memory in MB
            
        Returns:
            int: Estimated max batch size
        """
        if isinstance(self.gpu_manager, CPUDeviceManager):
            return self.batch_size # Return current batch size for CPU

        # This is a very rough heuristic. Needs to be calibrated.
        # Assume each item in a batch consumes roughly 10MB (example value)
        # This value should ideally come from profiling or configuration.
        memory_per_batch_item_mb = 10.0 # Placeholder, needs refinement
        if available_memory_mb > 0 and memory_per_batch_item_mb > 0:
            return max(1, int(available_memory_mb / memory_per_batch_item_mb))
        return 1

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Returns batch optimizer statistics.
        
        Returns:
            dict: Optimizer statistics
        """
        return self.optimizer_stats


_gpu_manager_instance: Optional[Union[JAXGPUManager, CPUDeviceManager]] = None

def get_gpu_manager() -> Union[JAXGPUManager, CPUDeviceManager]:
    """
    Returns a singleton instance of JAXGPUManager or CPUDeviceManager.

    Initializes the GPU manager if not already initialized. Ensures that JAX
    is configured for the appropriate device (GPU or CPU).
    """
    global _gpu_manager_instance
    if _gpu_manager_instance is None:
        if jax.device_count() > 0 and jax.devices()[0].platform in ('gpu', 'tpu'):
            _gpu_manager_instance = JAXGPUManager()
            _gpu_manager_instance.optimize_jax_config()
        else:
            _gpu_manager_instance = CPUDeviceManager()
            _gpu_manager_instance.optimize_jax_config()
    return _gpu_manager_instance

_batch_optimizer_instance: Optional[GPUBatchOptimizer] = None

def get_batch_optimizer() -> GPUBatchOptimizer:
    """
    Returns a singleton instance of GPUBatchOptimizer.
    """
    global _batch_optimizer_instance
    if _batch_optimizer_instance is None:
        # Ensure gpu_manager is initialized before creating batch optimizer
        manager = get_gpu_manager()
        _batch_optimizer_instance = GPUBatchOptimizer(manager)
    return _batch_optimizer_instance


def _init_gpu_optimization():
    """
    Initializes GPU optimization components like the GPU manager and batch optimizer.
    This function should be called once at application startup.
    """
    print("Initializing GPU optimization...")
    # Ensure the GPU manager is initialized
    get_gpu_manager()
    # Ensure the batch optimizer is initialized
    get_batch_optimizer()
    print("GPU optimization initialized.") 