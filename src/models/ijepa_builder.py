import torch
import timm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_ijepa_encoder(device=None):
    """
    Loads the official I-JEPA ViT-Huge model using timm.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    logging.info(f"Loading I-JEPA ViT-Huge model onto {device}...")
    
    # We use timm to avoid dynamic git clones and brittle tarball downloads
    model = timm.create_model('vit_huge_patch14_224.ijepa_in1k', pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Assign layer indices for our intervention hooks (matches notebook logic)
    for i, blk in enumerate(model.blocks):
        blk.layer_idx = i
        
    logging.info(f"Model ready. Total Layers: {len(model.blocks)}, Embed Dim: {model.embed_dim}")
    
    return model

if __name__ == "__main__":
    # Quick test to ensure it runs independently
    encoder = build_ijepa_encoder()

