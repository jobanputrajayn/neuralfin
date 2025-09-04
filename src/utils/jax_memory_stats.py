import jax
import jax.profiler

def print_jax_memory_stats():
    devices = jax.devices()
    for i, device in enumerate(devices):
        print(f"\nDevice {i}: {device}")
        try:
            stats = device.memory_stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  Could not retrieve memory stats: {e}")

if __name__ == "__main__":
    print("=== JAX Device Memory Stats ===")
    print_jax_memory_stats() 
    jax.profiler.save_device_memory_profile("memory.prof")