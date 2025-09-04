import inspect
import io
from contextlib import redirect_stdout
from flax import nnx
import re

def extract_nnx_structure(model, parent_name="", depth=0):
    """
    Extract the structure of an nnx.Module by fully walking its __nnx_repr__ tree.
    For every node (module, param, array, etc.), attempts to extract a shape using a generalized approach.
    Returns a list of dicts: [{name, type, shape, depth, parent}]
    """
    from flax import nnx
    def get_layer_shape(obj):
        # Special case for LIF: use mem_pot.value.shape
        if type(obj).__name__ == "LIF" and hasattr(obj, "mem_pot") and hasattr(obj.mem_pot, "value"):
            if hasattr(obj.mem_pot.value, "shape"):
                return tuple(obj.mem_pot.value.shape)
        # Try to get shape for known layer types (Linear, Conv, etc.)
        for attr in ["shape", "kernel", "weight", "W"]:
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                if hasattr(v, "shape"):
                    return tuple(v.shape)
        # Fallback: look for .out_features, .features, .num_features
        for attr in ["out_features", "features", "num_features"]:
            if hasattr(obj, attr):
                return getattr(obj, attr)
        return None
    def walk_user_layers(obj, name, parent, depth):
        layers = []
        if isinstance(obj, nnx.Module):
            layers.append({
                "name": name,
                "type": type(obj).__name__,
                "depth": depth,
                "parent": parent,
                "shape": get_layer_shape(obj)
            })
        it = obj.__nnx_repr__() if hasattr(obj, "__nnx_repr__") else None
        if it is not None:
            try:
                next(it)  # skip Object
            except Exception:
                pass
            for attr in it:
                val = attr.value
                child_name = attr.key
                if isinstance(val, (list, tuple)):
                    for i, v in enumerate(val):
                        layers.extend(walk_user_layers(v, f"{child_name}[{i}]", name, depth+1))
                else:
                    layers.extend(walk_user_layers(val, child_name, name, depth+1))
        return layers
    return walk_user_layers(model, parent_name or type(model).__name__, parent_name if depth > 0 else "", depth)