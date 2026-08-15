import os
import torch
import pandas as pd
import logging
import yaml

from src.models.ijepa_builder import build_ijepa_encoder
from src.data.dataset_generator import generate_counterfactual_batch
from src.experiments.patching import cache_activations, PatchLayerContext
from src.utils.metrics import get_feature_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # 1. Load config
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Model & Data
    encoder = build_ijepa_encoder(device)
    x_plus, x_minus = generate_counterfactual_batch(batch_size=config['experiment']['batch_size'], device=device)
    
    # 3. Compute Feature Direction
    logging.info("Computing feature direction...")
    with torch.no_grad():
        out_plus = encoder(x_plus)
        out_minus = encoder(x_minus)
        
    emb_plus = out_plus.mean(dim=1)
    emb_minus = out_minus.mean(dim=1)
    feature_dir = (emb_plus.mean(dim=0) - emb_minus.mean(dim=0))
    feature_dir = feature_dir / feature_dir.norm()
    
    # 4. Cache Counterfactual Activations
    minus_cache = cache_activations(encoder, x_minus, device)
    
    # 5. Run Patching Scan
    with torch.no_grad():
        clean_score = get_feature_score(encoder(x_plus), feature_dir)
    logging.info(f"Clean Score: {clean_score:.4f}")
    
    results = []
    num_layers = len(encoder.blocks)
    
    for i in range(num_layers):
        target_tensor = minus_cache[f'layer_{i}']
        
        # Safely patch the layer using our Context Manager
        with PatchLayerContext(encoder.blocks[i], target_tensor):
            with torch.no_grad():
                patched_out = encoder(x_plus)
                patched_score = get_feature_score(patched_out, feature_dir)
                
        effect = clean_score - patched_score
        results.append({'layer': i, 'clean': clean_score, 'patched': patched_score, 'effect': effect})
        logging.info(f"Layer {i:2d} | Patched Score: {patched_score:.4f} | Causal Effect: {effect:.4f}")
        
    # 6. Save Results
    df = pd.DataFrame(results)
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    save_path = os.path.join(config['output']['save_dir'], 'results.csv')
    df.to_csv(save_path, index=False)
    logging.info(f"Experiment complete! Results saved to {save_path}")

if __name__ == "__main__":
    main()