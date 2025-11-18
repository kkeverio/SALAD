"""Utilities for relation modeling on composition maps.

This module provides:
    * dataclass :class:`Component` describing connected components that appear in
      per-pixel class predictions / annotations.
    * helpers to extract connected components from class maps created by the
      SALAD composition networks (or any other segmentation model).
    * pair feature builders and synthetic anomaly utilities that are generic
      enough to be re-used across tasks.
    * :class:`PairRelationModel`, a light-weight multi-layer perceptron that can
      score pairs of components.

Assumptions
-----------
``class_map`` tensors are expected to be shaped ``(H, W)`` or ``(1, H, W)`` and
contain integer class identifiers in ``[0, num_classes - 1]`` where ``0`` is
typically used for the background.  When batched data is handled, the code
expects a Python sequence of class maps (one per image) rather than a
``torch.Tensor`` batched tensor; this makes the utilities agnostic to how data
loading is implemented and easier to integrate in custom training loops.

All components operate on CPU tensors by default but the exported helper
functions expose a ``device`` argument so that features can be created on the
same device as the model without manual ``to(device)`` calls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage
from torch import Tensor, nn


Pair = Tuple["Component", "Component"]


@dataclass
class Component:
    """Describes a connected component coming from a composition map.

    Attributes
    ----------
    class_id:
        Integer class identifier (starting at zero) that the component belongs
        to.
    mask:
        Boolean tensor of shape ``(H, W)`` containing the pixels that belong to
        this component.  The tensor is stored on CPU to simplify I/O but can be
        moved to any device when needed.
    bbox:
        Tensor of shape ``(4,)`` formatted as ``(x1, y1, x2, y2)`` in pixels.
    centroid:
        Tensor of shape ``(2,)`` storing ``(cx, cy)`` measured in pixels.
    area:
        Float indicating the number of foreground pixels.
    image_index:
        Optional index of the image this component originated from.  This is
        extremely useful for building cross-image pairs when synthesising
        anomalies.
    tag:
        Optional tag that can be filled with arbitrary meta-data (ground-truth
        label, confidence, etc.).
    """

    class_id: int
    mask: Tensor
    bbox: Tensor
    centroid: Tensor
    area: float
    image_index: Optional[int] = None
    tag: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mask.dtype != torch.bool:
            raise TypeError("Component.mask must be a boolean tensor")
        if self.mask.ndim != 2:
            raise ValueError("Component.mask is expected to be a 2-D tensor")

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def aspect_ratio(self) -> float:
        h = max(self.height, 1e-3)
        return self.width / h

    def iou(self, other: "Component") -> float:
        """Intersection over union computed on binary masks."""

        if self.mask.shape != other.mask.shape:
            raise ValueError("Components must share the same spatial size")
        inter = torch.logical_and(self.mask, other.mask).sum().item()
        union = torch.logical_or(self.mask, other.mask).sum().item()
        if union == 0:
            return 0.0
        return float(inter / union)

    def copy_with(self, **kwargs) -> "Component":
        """Shallow copy helper used by synthetic anomaly generators."""

        return replace(self, **kwargs)


def _to_hw(class_map: Tensor) -> Tensor:
    """Ensures the class map has shape ``(H, W)``."""

    if class_map.ndim == 3 and class_map.shape[0] == 1:
        class_map = class_map.squeeze(0)
    if class_map.ndim != 2:
        raise ValueError(
            "Class maps must have shape (H, W) or (1, H, W); received"
            f" {tuple(class_map.shape)}"
        )
    return class_map


def extract_components_from_class_map(
    class_map: Tensor,
    background_class: int = 0,
    min_area: int = 10,
    connectivity: int = 1,
    image_index: Optional[int] = None,
) -> List[Component]:
    """Extract connected components from a per-pixel class map.

    Parameters
    ----------
    class_map:
        Tensor of shape ``(H, W)`` or ``(1, H, W)`` containing integer class
        identifiers.
    background_class:
        Component extraction ignores pixels with this class id.
    min_area:
        Components with fewer pixels than ``min_area`` are discarded.
    connectivity:
        Passed to :func:`scipy.ndimage.label` to control how diagonal pixels are
        connected.
    image_index:
        Optional origin index to attach to resulting components.
    """

    class_map = _to_hw(class_map).to(torch.int64)
    np_map = class_map.cpu().numpy()
    components: List[Component] = []
    height, width = class_map.shape

    for class_id in np.unique(np_map):
        if class_id == background_class:
            continue
        mask = np_map == class_id
        labeled, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, connectivity))
        if count == 0:
            continue
        for label_idx in range(1, count + 1):
            component_mask = labeled == label_idx
            area = int(component_mask.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(component_mask)
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            centroid = torch.tensor([xs.mean(), ys.mean()], dtype=torch.float32)
            tensor_mask = torch.from_numpy(component_mask).to(torch.bool)
            component = Component(
                class_id=int(class_id),
                mask=tensor_mask,
                bbox=bbox,
                centroid=centroid,
                area=float(area),
                image_index=image_index,
            )
            components.append(component)

    return components


def components_to_boxes(components: Sequence[Component]) -> Tensor:
    """Returns a tensor of bounding boxes for convenience."""

    if not components:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.stack([c.bbox for c in components], dim=0)


def _compute_geometric_features(component_a: Component, component_b: Component) -> Tensor:
    """Computes geometry descriptors for a pair of components."""

    dx = component_b.centroid[0] - component_a.centroid[0]
    dy = component_b.centroid[1] - component_a.centroid[1]
    distance = torch.sqrt(dx**2 + dy**2 + 1e-6)
    log_area_a = torch.log(torch.tensor(component_a.area + 1.0))
    log_area_b = torch.log(torch.tensor(component_b.area + 1.0))
    area_ratio = torch.log(torch.tensor((component_a.area + 1.0) / (component_b.area + 1.0)))
    aspect_a = torch.tensor(component_a.aspect_ratio)
    aspect_b = torch.tensor(component_b.aspect_ratio)
    iou = torch.tensor(component_a.iou(component_b))
    geom = torch.stack(
        [dx, dy, distance, log_area_a, log_area_b, area_ratio, aspect_a, aspect_b, iou]
    )
    return geom.float()


def build_pair_feature_tensor(
    pairs: Sequence[Pair],
    class_embedder: nn.Embedding,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Embeds class ids and concatenates them with geometric features."""

    if device is None:
        device = class_embedder.weight.device

    if not pairs:
        embedding_dim = class_embedder.embedding_dim * 2
        geom_dim = 9
        return torch.zeros((0, embedding_dim + geom_dim), device=device)

    class_ids_a = torch.tensor([a.class_id for a, _ in pairs], dtype=torch.long, device=device)
    class_ids_b = torch.tensor([b.class_id for _, b in pairs], dtype=torch.long, device=device)
    emb_a = class_embedder(class_ids_a)
    emb_b = class_embedder(class_ids_b)

    geom_features = torch.stack([
        _compute_geometric_features(a, b) for a, b in pairs
    ]).to(device)
    features = torch.cat([emb_a, emb_b, geom_features], dim=-1)
    return features


def corrupt_class_ids(
    components: Sequence[Component],
    num_classes: int,
    corruption_prob: float = 0.5,
    rng: Optional[random.Random] = None,
) -> List[Component]:
    """Returns new components with randomly corrupted class identifiers."""

    rng = rng or random.Random()
    corrupted: List[Component] = []
    for component in components:
        if rng.random() >= corruption_prob:
            continue
        new_class = component.class_id
        while new_class == component.class_id:
            new_class = rng.randrange(num_classes)
        corrupted.append(component.copy_with(class_id=new_class))
    return corrupted


def corrupt_geometry(
    component: Component,
    jitter_std: float = 0.1,
    rng: Optional[random.Random] = None,
) -> Component:
    """Returns a shallow copy with jittered centroid/bbox."""

    rng = rng or random.Random()
    jitter = torch.tensor([rng.gauss(0.0, jitter_std), rng.gauss(0.0, jitter_std)])
    centroid = component.centroid + jitter
    bbox = component.bbox + torch.tensor([jitter[0], jitter[1], jitter[0], jitter[1]])
    return component.copy_with(centroid=centroid, bbox=bbox)


def mix_cross_image_components(components: Sequence[Component], rng: Optional[random.Random] = None) -> List[Pair]:
    """Builds component pairs sampled from different source images."""

    rng = rng or random.Random()
    by_image = {}
    for component in components:
        by_image.setdefault(component.image_index, []).append(component)
    image_ids = [idx for idx in by_image.keys() if idx is not None]
    if len(image_ids) < 2:
        return []

    pairs: List[Pair] = []
    for _ in range(len(components)):
        img_a, img_b = rng.sample(image_ids, 2)
        comp_a = rng.choice(by_image[img_a])
        comp_b = rng.choice(by_image[img_b])
        pairs.append((comp_a, comp_b))
    return pairs


def generate_negative_pairs(
    components_by_image: Sequence[Sequence[Component]],
    num_classes: int,
    desired_count: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Pair]:
    """Generates synthetic negative pairs mixing several corruption strategies."""

    rng = rng or random.Random()
    all_components = [c for comps in components_by_image for c in comps]
    if desired_count is None:
        desired_count = max(len(all_components), 1)
    if len(all_components) < 2:
        return []

    negatives: List[Pair] = []
    while len(negatives) < desired_count:
        strategy = rng.choice(["cross", "class", "geom"])
        if strategy == "cross":
            pairs = mix_cross_image_components(all_components, rng)
            negatives.extend(pairs)
        elif strategy == "class":
            base = rng.choice(all_components)
            corrupted = corrupt_class_ids([base], num_classes, corruption_prob=1.0, rng=rng)
            if corrupted:
                partner = rng.choice(all_components)
                negatives.append((corrupted[0], partner))
        else:  # geom
            comp_a, comp_b = rng.sample(all_components, 2)
            negatives.append((corrupt_geometry(comp_a, rng=rng), comp_b))
        negatives = negatives[:desired_count]
    return negatives


def aggregate_pair_scores(scores: Tensor, reduction: str = "mean") -> Tensor:
    """Aggregates per-pair scores using a simple reduction strategy."""

    if scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device)
    if reduction == "mean":
        return scores.mean()
    if reduction == "max":
        return scores.max()
    if reduction == "sum":
        return scores.sum()
    raise ValueError(f"Unknown reduction '{reduction}'")


class PairRelationModel(nn.Module):
    """Simple MLP scoring the compatibility between two components."""

    def __init__(
        self,
        num_classes: int,
        class_embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        geom_dim = 9
        input_dim = class_embedding_dim * 2 + geom_dim
        layers: List[nn.Module] = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """Computes logits from already-built feature vectors."""

        return self.mlp(features).squeeze(-1)

    def forward_pairs(
        self,
        pairs: Sequence[Pair],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Builds features from :class:`Component` pairs and runs the MLP."""

        features = build_pair_feature_tensor(pairs, self.class_embedding, device)
        return self.forward(features)

    def build_training_batch(
        self,
        components_by_image: Sequence[Sequence[Component]],
        negative_ratio: float = 1.0,
        device: Optional[torch.device] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Constructs feature/label tensors for supervised training.

        Positive pairs are created from all combinations within the same image.
        Synthetic negatives are sampled using :func:`generate_negative_pairs`.
        """

        rng = rng or random.Random()
        positives: List[Pair] = []
        for components in components_by_image:
            positives.extend(list(combinations(components, 2)))
        desired_negatives = int(len(positives) * negative_ratio)
        negatives = generate_negative_pairs(
            components_by_image,
            num_classes=self.num_classes,
            desired_count=max(desired_negatives, 1) if desired_negatives else 0,
            rng=rng,
        )
        all_pairs = positives + negatives
        labels = torch.tensor(
            [1.0] * len(positives) + [0.0] * len(negatives),
            dtype=torch.float32,
            device=device or self.class_embedding.weight.device,
        )
        features = build_pair_feature_tensor(all_pairs, self.class_embedding, device)
        return features, labels

    @torch.no_grad()
    def score_image(
        self,
        components: Sequence[Component],
        reduction: str = "mean",
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Scores all in-image pairs and aggregates the scores.

        Returns the per-pair probabilities (after sigmoid) together with a
        scalar aggregated score according to the chosen reduction.
        """

        pairs = list(combinations(components, 2))
        if not pairs:
            zero = torch.tensor(0.0, device=self.class_embedding.weight.device)
            return zero, zero
        logits = self.forward_pairs(pairs, device=device)
        probs = torch.sigmoid(logits)
        agg = aggregate_pair_scores(probs, reduction=reduction)
        return probs, agg


__all__ = [
    "Component",
    "Pair",
    "PairRelationModel",
    "aggregate_pair_scores",
    "build_pair_feature_tensor",
    "components_to_boxes",
    "corrupt_class_ids",
    "corrupt_geometry",
    "extract_components_from_class_map",
    "generate_negative_pairs",
    "mix_cross_image_components",
]

