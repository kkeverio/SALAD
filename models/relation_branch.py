"""Utilities for relation-branch inference.

The relation branch reasons about the spatial relationships between
components on a composition map.  At inference time we need to extract the
connected components, form features for all unique pairs, run the trained
relation model and aggregate the resulting anomaly scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label


@dataclass
class Component:
    """Light-weight representation of a connected component."""

    class_id: int
    area: float
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]


def _prepare_segmentation(segmentation: torch.Tensor) -> torch.Tensor:
    """Ensure segmentation tensors are 3D (C, H, W)."""

    if segmentation.dim() == 4:
        segmentation = segmentation.squeeze(0)
    if segmentation.dim() == 2:
        segmentation = segmentation.unsqueeze(0)
    return segmentation


def _extract_components(
    segmentation: torch.Tensor,
    min_component_pixels: int = 32,
) -> List[Component]:
    """Extract connected components from a segmentation tensor.

    Args:
        segmentation: Tensor of shape (C, H, W) containing one-hot masks.
        min_component_pixels: Minimum number of pixels for a component to be
            considered valid.  This removes tiny artifacts coming from the
            segmentation network.

    Returns:
        List of :class:`Component` instances.
    """

    segmentation = _prepare_segmentation(segmentation)
    seg_np = segmentation.detach().float().cpu().numpy()
    _, height, width = segmentation.shape
    components: List[Component] = []

    total_pixels = float(height * width)
    for class_id, mask in enumerate(seg_np):
        binary_mask = mask > 0.5
        if not np.any(binary_mask):
            continue
        labeled_mask, num_features = label(binary_mask)
        for label_id in range(1, num_features + 1):
            component_mask = labeled_mask == label_id
            area = float(component_mask.sum())
            if area < min_component_pixels:
                continue
            coords = np.argwhere(component_mask)
            if coords.size == 0:
                continue
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            centroid_y, centroid_x = coords.mean(axis=0)
            components.append(
                Component(
                    class_id=class_id,
                    area=area / total_pixels,
                    centroid=(float(centroid_x / width), float(centroid_y / height)),
                    bbox=(
                        float(min_x / width),
                        float(min_y / height),
                        float(max_x / width),
                        float(max_y / height),
                    ),
                )
            )

    return components


def _pairwise_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    intersection = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _build_pair_features(
    components: Sequence[Component],
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, Component]]]:
    features: List[torch.Tensor] = []
    metadata: List[Dict[str, Component]] = []

    for idx_a, comp_a in enumerate(components):
        for idx_b in range(idx_a + 1, len(components)):
            comp_b = components[idx_b]
            class_feat_a = F.one_hot(torch.tensor(comp_a.class_id), num_classes=num_classes)
            class_feat_b = F.one_hot(torch.tensor(comp_b.class_id), num_classes=num_classes)
            class_feat = torch.cat([class_feat_a, class_feat_b], dim=0).float()

            delta_x = comp_a.centroid[0] - comp_b.centroid[0]
            delta_y = comp_a.centroid[1] - comp_b.centroid[1]
            euclidean = float(np.sqrt(delta_x**2 + delta_y**2))
            iou = _pairwise_iou(comp_a.bbox, comp_b.bbox)

            geom_feat = torch.tensor(
                [
                    comp_a.area,
                    comp_b.area,
                    comp_a.centroid[0],
                    comp_a.centroid[1],
                    comp_b.centroid[0],
                    comp_b.centroid[1],
                    delta_x,
                    delta_y,
                    abs(delta_x),
                    abs(delta_y),
                    euclidean,
                    iou,
                ],
                dtype=torch.float32,
            )

            features.append(torch.cat([class_feat, geom_feat], dim=0))
            metadata.append({"component_a": comp_a, "component_b": comp_b})

    if not features:
        return torch.empty(0, 0, device=device), metadata

    feature_tensor = torch.stack(features).to(device)
    return feature_tensor, metadata


def compute_relation_anomaly_score(
    segmentation: torch.Tensor,
    relation_model: torch.nn.Module,
    aggregation: str = "topk",
    topk: int = 3,
    min_component_pixels: int = 32,
) -> Dict[str, object]:
    """Run the relation model and return pairwise and image-level scores."""

    if relation_model is None:
        return {"image_score": 0.0, "pair_scores": [], "top_pairs": []}

    segmentation = _prepare_segmentation(segmentation)
    components = _extract_components(segmentation, min_component_pixels=min_component_pixels)
    if len(components) < 2:
        return {"image_score": 0.0, "pair_scores": [], "top_pairs": []}

    try:
        device = next(relation_model.parameters()).device
    except StopIteration:
        device = segmentation.device
    features, metadata = _build_pair_features(components, segmentation.shape[0], device=device)
    if features.numel() == 0:
        return {"image_score": 0.0, "pair_scores": [], "top_pairs": []}

    relation_model.eval()
    with torch.no_grad():
        scores = relation_model(features)
        if isinstance(scores, (tuple, list)):
            scores = scores[0]
        scores = scores.squeeze(-1)

    if scores.ndim == 0:
        scores = scores.unsqueeze(0)

    scores_cpu = scores.detach().cpu().float()
    pair_scores: List[Dict[str, object]] = []
    for score, meta in zip(scores_cpu.tolist(), metadata):
        pair_scores.append(
            {
                "score": float(score),
                "component_a": meta["component_a"],
                "component_b": meta["component_b"],
            }
        )

    pair_scores.sort(key=lambda item: item["score"], reverse=True)

    if scores_cpu.numel() == 0:
        image_score = 0.0
    elif aggregation == "max":
        image_score = float(scores_cpu.max().item())
    else:
        k = max(1, min(int(topk), scores_cpu.numel()))
        topk_values = torch.topk(scores_cpu, k=k).values
        image_score = float(topk_values.mean().item())

    top_pairs = pair_scores[: max(1, min(int(topk), len(pair_scores)))]
    return {"image_score": image_score, "pair_scores": pair_scores, "top_pairs": top_pairs}


__all__ = ["compute_relation_anomaly_score", "Component"]

