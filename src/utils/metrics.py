import torch
import torch.nn.functional as F

def get_feature_score(embeddings, feat_dir):
    """
    Computes how strongly the embeddings align with the target feature direction.
    """
    pooled = embeddings.mean(dim=1)
    pooled = F.normalize(pooled, dim=-1)
    return torch.matmul(pooled, feat_dir).mean().item()