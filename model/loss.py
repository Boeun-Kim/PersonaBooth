import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(features, temperature=0.5):
    """
    NT-Xent loss (Normalized Temperature-scaled Cross Entropy Loss) implementation.

    Args:
        features: torch.Tensor of shape [2N, D] where N is the batch size and D is the feature dimension.
                  The first N samples correspond to the positive pairs of the last N samples.
        temperature: temperature scaling parameter (tau).
        
    Returns:
        torch.Tensor: The computed NT-Xent loss.
    """
    
    # Normalize the features to unit vectors
    features = F.normalize(features, dim=1)
    
    # Compute cosine similarity between all pairs
    similarity_matrix = torch.matmul(features, features.T)  # [2N, 2N]
    similarity_matrix = similarity_matrix / temperature
    
    # Create labels: each positive pair (i, j) is located at (i, N+i) and (N+i, i)
    batch_size = features.shape[0] // 2
    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(features.device)

    # Mask to ignore the diagonal similarity (self-similarity)
    mask = torch.eye(2 * batch_size, device=features.device).bool()
    
    # Exclude self-similarities by masking out the diagonal
    logits = similarity_matrix.masked_fill(mask, float('-inf'))
    
    # Compute the loss using cross entropy
    loss = F.cross_entropy(logits, labels)
    
    return loss