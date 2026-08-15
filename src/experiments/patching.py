import torch
import logging

class PatchLayerContext:
    """
    A safe context manager for patching PyTorch module activations.
    Guarantees that the forward hook is removed upon exiting the block,
    even if an exception is raised.
    """
    def __init__(self, module, target_tensor):
        self.module = module
        self.target_tensor = target_tensor
        self.handle = None

    def __enter__(self):
        def patch_hook(mod, inp, out):
            return self.target_tensor
        
        # Register the hook on entry
        self.handle = self.module.register_forward_hook(patch_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Guarantee removal on exit
        if self.handle is not None:
            self.handle.remove()


def cache_activations(model, x, device=None):
    """
    Runs a forward pass and caches the output activations of every block.
    Matches the 'minus_cache' logic from the baseline.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    cache = {}
    handles = []

    def make_cache_hook(name):
        def hook(module, inp, out):
            cache[name] = out.detach() # Keep on GPU
        return hook

    # Attach hooks to all blocks
    for i, blk in enumerate(model.blocks):
        h = blk.register_forward_hook(make_cache_hook(f'layer_{i}'))
        handles.append(h)

    logging.info("Caching counterfactual activations...")
    with torch.no_grad():
        _ = model(x.to(device))

    # Clean up caching hooks
    for h in handles:
        h.remove()
        
    logging.info(f"Cached activations for {len(cache)} layers.")
    return cache

