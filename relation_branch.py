import itertools
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RelationClassEmbedding(nn.Module):
    """Learns embeddings for composition classes."""

    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, class_pairs: torch.Tensor) -> torch.Tensor:
        """Embeds a batch of class index pairs.

        Args:
            class_pairs: Tensor of shape (N, 2) containing class indices.

        Returns:
            Tensor of shape (N, 2 * embedding_dim) containing flattened
            embeddings for each class pair.
        """
        if class_pairs.numel() == 0:
            return class_pairs.new_zeros((0, self.embedding.embedding_dim * 2))
        embedded = self.embedding(class_pairs.long())
        return embedded.view(embedded.shape[0], -1)


class PairRelationModel(nn.Module):
    """Simple MLP that predicts whether a pair of classes is compatible."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = embedding_dim * 2
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.model(pair_features).squeeze(-1)


def _collect_pairs_from_presence(presence: torch.Tensor) -> list:
    """Collects unique unordered class index pairs from a presence mask."""
    pairs = []
    class_indices = torch.nonzero(presence, as_tuple=False).view(-1).tolist()
    if len(class_indices) < 2:
        return pairs
    for i, j in itertools.combinations(class_indices, 2):
        pairs.append((i, j))
    return pairs


def build_relation_pair_batch(
    seg_maps: torch.Tensor,
    anom_seg_maps: torch.Tensor,
    max_pairs: int,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Builds positive/negative class pairs from clean/anomalous maps.

    Args:
        seg_maps: Clean composition maps with shape (B, C, H, W).
        anom_seg_maps: Anomalous composition maps with shape (B, C, H, W).
        max_pairs: Maximum number of positive/negative pairs to sample.
        device: Target device for the returned tensors.

    Returns:
        A tuple of (pair_indices, pair_labels). pair_indices has shape (N, 2)
        and pair_labels has shape (N,), or (None, None) if no pairs are found.
    """
    if seg_maps is None or anom_seg_maps is None:
        return None, None

    if seg_maps.dim() == 3:
        seg_maps = seg_maps.unsqueeze(0)
    if anom_seg_maps.dim() == 3:
        anom_seg_maps = anom_seg_maps.unsqueeze(0)

    if device is None:
        device = seg_maps.device

    with torch.no_grad():
        seg_presence = (seg_maps.sum(dim=(-1, -2)) > 0).cpu()
        anom_presence = (anom_seg_maps.sum(dim=(-1, -2)) > 0).cpu()

    pos_pairs = []
    for sample_presence in seg_presence:
        pos_pairs.extend(_collect_pairs_from_presence(sample_presence))

    neg_pairs = []
    for sample_presence in anom_presence:
        neg_pairs.extend(_collect_pairs_from_presence(sample_presence))

    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)

    if max_pairs > 0:
        pos_pairs = pos_pairs[:max_pairs]
        neg_pairs = neg_pairs[:max_pairs]

    # Ensure we always have negatives by synthesizing random ones when needed.
    num_classes = seg_maps.shape[1]
    while len(neg_pairs) < len(pos_pairs) and num_classes > 1:
        candidate = tuple(random.sample(range(num_classes), 2))
        neg_pairs.append(candidate)

    all_pairs = pos_pairs + neg_pairs
    if not all_pairs:
        return None, None

    labels = [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)
    pair_indices = torch.tensor(all_pairs, dtype=torch.long, device=device)
    pair_labels = torch.tensor(labels, dtype=torch.float32, device=device)
    return pair_indices, pair_labels
