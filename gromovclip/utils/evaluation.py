"""
Read-outs shared by the evaluators.

A zero-shot classifier is built from several prompts per class, and there are
two ways to ensemble them. *Feature-level* ensembling averages the prompts in
the tangent space and lifts the mean, so a class is one point. *Score-level*
ensembling lifts every prompt on its own and scores a class by the mean
distance of its prompts to the image.

Each read-out also exists without the exponential map: the encoder outputs are
scaled by the learned alpha and compared by Euclidean distance. The functions
for it carry the suffix `_tangent`.
"""

import torch
from torch.nn import functional as F

from gromovclip import euclidean as E
from gromovclip import lorentz as L
from gromovclip.models import (
    MERU,
    CLIPBaseline,
    HyCoCLIP,
    PHyCLIP,
)


def _product_curvatures(model: PHyCLIP) -> torch.Tensor:
    return torch.stack([curv.exp() for curv in model.curvs])


def _combine_factors(model: PHyCLIP, distances: torch.Tensor) -> torch.Tensor:
    """Combine per-factor distances `(N, B1, B2)` with the model's product metric."""
    if model.product_metric == "l1":
        return distances.mean(dim=0)
    return torch.sqrt((distances**2).mean(dim=0))


def process_text_features(
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Feature-level classifier of one class.

    Args:
        model: The model instance
        text_features: Raw per-prompt text features, `project=False`.

    Returns:
        One classifier vector, ready for `compute_similarity_scores`.
    """
    if isinstance(model, PHyCLIP):
        # map class features to each concept's hyperboloid
        class_feats = text_features.mean(dim=0).view(
            -1, model.num_subspaces, model.subspace_dim
        )

        class_feats = class_feats * torch.stack(
            [alpha.exp() for alpha in model.textual_concept_alphas]
        ).view(1, -1, 1)

        curvs = _product_curvatures(model)
        class_feats = L.exp_map0_batch(class_feats, curvs).view(model.num_subspaces, -1)
        return class_feats
    elif isinstance(model, (HyCoCLIP, MERU)):
        # Ensemble in the tangent space, then project to Hyperboloid.
        class_feats = text_features.mean(dim=0)
        class_feats = class_feats * model.textual_alpha.exp()
        class_feats = L.exp_map0(class_feats, model.curv.exp())
        return class_feats
    else:
        # Ensemble prompt features: normalize -> average -> normalize.
        class_feats = F.normalize(text_features, dim=-1)
        class_feats = class_feats.mean(dim=0)
        class_feats = F.normalize(class_feats, dim=-1)
        return class_feats


def compute_similarity_scores(
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute similarity scores between image and text features based on model type.

    Args:
        model: The model instance
        image_features: Image features
        text_features: Processed text features (classifier)

    Returns:
        Similarity scores
    """

    if isinstance(model, PHyCLIP):
        curvs = _product_curvatures(model)
        image_features = image_features.view(
            -1, model.num_subspaces, model.subspace_dim
        )
        text_features = text_features.view(-1, model.num_subspaces, model.subspace_dim)
        scores = L.pairwise_inner_batch(image_features, text_features, curvs).sum(dim=0)
        return scores
    elif isinstance(model, (HyCoCLIP, MERU)):
        scores = L.pairwise_inner(image_features, text_features, model.curv.exp())
        return scores
    else:
        scores = image_features @ text_features.T
        return scores


def score_level_classifier(
    model: PHyCLIP | MERU,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Score-level classifier of one class: every prompt lifted on its own, with
    the prompt axis kept.

    Args:
        model: A MERU subclass. `PHyCLIP` lifts onto its product of
            hyperboloids.
        text_features: Raw per-prompt text features, `project=False`, shape
            `(num_prompts, embed_dim)`.

    Returns:
        Points of shape `(num_prompts, embed_dim)`, or
        `(num_prompts, num_subspaces, subspace_dim)` for `PHyCLIP`.
    """
    if isinstance(model, PHyCLIP):
        class_feats = text_features.view(
            -1, model.num_subspaces, model.subspace_dim
        )
        class_feats = class_feats * torch.stack(
            [alpha.exp() for alpha in model.textual_concept_alphas]
        ).view(1, -1, 1)
        curvs = _product_curvatures(model)
        return L.exp_map0_batch(class_feats, curvs, eps=model.lorentz_eps)

    class_feats = text_features * model.textual_alpha.exp()
    return L.exp_map0(class_feats, model.curv.exp(), eps=model.lorentz_eps)


def score_level_scores(
    model: PHyCLIP | MERU,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Negative distance of each image to each prompt of each class, averaged over
    the prompts of the class. The distance is the one in the model's
    contrastive loss: geodesic for a single hyperboloid, per-factor geodesic
    combined by the product metric for `PHyCLIP`.

    Args:
        image_features: Image features of shape `(B, embed_dim)`.
        text_features: Classifier of shape `(num_classes, num_prompts, embed_dim)`,
            or `(num_classes, num_prompts, num_subspaces, subspace_dim)` for
            `PHyCLIP`.
    """
    if isinstance(model, PHyCLIP):
        num_classes, num_prompts = text_features.shape[:2]
        distances = L.pairwise_dist_batch(
            image_features.view(-1, model.num_subspaces, model.subspace_dim),
            text_features.reshape(
                num_classes * num_prompts, model.num_subspaces, model.subspace_dim
            ),
            _product_curvatures(model),
            eps=model.lorentz_eps,
        )
        combined = _combine_factors(model, distances)
        return -combined.view(-1, num_classes, num_prompts).mean(dim=-1)

    num_classes, num_prompts, embed_dim = text_features.shape
    distances = L.pairwise_dist(
        image_features,
        text_features.reshape(num_classes * num_prompts, embed_dim),
        model.curv.exp(),
        eps=model.lorentz_eps,
    )
    return -distances.view(-1, num_classes, num_prompts).mean(dim=-1)


def to_tangent_features(
    model: PHyCLIP | HyCoCLIP | MERU,
    feats: torch.Tensor,
    modality: str,
) -> torch.Tensor:
    """
    Scale raw (`project=False`) encoder features by the learned alpha, as is
    done before `exp_map0`, and skip the exponential map.

    Args:
        model: A MERU subclass.
        feats: Output of `model.encode_{image,text}(..., project=False)`.
        modality: Either `"image"` or `"text"`.
    """
    if isinstance(model, PHyCLIP):
        alphas = (
            model.visual_concept_alphas
            if modality == "image"
            else model.textual_concept_alphas
        )
        feats = feats.view(-1, model.num_subspaces, model.subspace_dim)
        feats = feats * torch.stack([a.exp() for a in alphas]).view(1, -1, 1)
        return feats.view(-1, model.embed_dim)

    alpha = model.visual_alpha if modality == "image" else model.textual_alpha
    return feats * alpha.exp()


def process_text_features_tangent(
    model: PHyCLIP | HyCoCLIP | MERU,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Feature-level classifier of one class without the exponential map: the
    prompts are averaged and scaled by the learned alpha. The scaling is
    linear, so it commutes with the average.

    Args:
        model: A MERU subclass.
        text_features: Raw per-prompt text features, `project=False`.
    """
    if isinstance(model, PHyCLIP):
        class_feats = text_features.mean(dim=0).view(
            -1, model.num_subspaces, model.subspace_dim
        )
        class_feats = class_feats * torch.stack(
            [alpha.exp() for alpha in model.textual_concept_alphas]
        ).view(1, -1, 1)
        return class_feats.view(model.num_subspaces, -1)
    else:
        class_feats = text_features.mean(dim=0)
        return class_feats * model.textual_alpha.exp()


def compute_tangent_similarity_scores(
    model: PHyCLIP | HyCoCLIP | MERU,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Negative Euclidean distance between two batches of tangent-space features
    (see `to_tangent_features`). For `PHyCLIP` the distance is taken per factor
    and combined with the model's product metric.
    """
    if isinstance(model, PHyCLIP):
        image_features = image_features.view(
            -1, model.num_subspaces, model.subspace_dim
        )
        text_features = text_features.view(-1, model.num_subspaces, model.subspace_dim)
        distances = E.pairwise_dist_batch(
            image_features, text_features, eps=model.lorentz_eps
        )
        return -_combine_factors(model, distances)

    return -E.pairwise_dist(image_features, text_features)


def score_level_classifier_tangent(
    model: PHyCLIP | MERU,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Score-level classifier of one class without the exponential map, shape
    `(num_prompts, embed_dim)`.
    """
    if isinstance(model, PHyCLIP):
        return to_tangent_features(model, text_features, "text")
    return text_features * model.textual_alpha.exp()


def score_level_scores_tangent(
    model: PHyCLIP | MERU,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Negative Euclidean distance of each image to each prompt in the tangent
    space, averaged over the prompts of the class.

    Args:
        image_features: Tangent-space image features of shape `(B, embed_dim)`.
        text_features: Classifier of shape `(num_classes, num_prompts, embed_dim)`.
    """
    num_classes, num_prompts, embed_dim = text_features.shape
    text_features = text_features.reshape(num_classes * num_prompts, embed_dim)
    if isinstance(model, PHyCLIP):
        scores = compute_tangent_similarity_scores(model, image_features, text_features)
        return scores.view(-1, num_classes, num_prompts).mean(dim=-1)
    distances = E.pairwise_dist(image_features, text_features, eps=model.lorentz_eps)
    return -distances.view(-1, num_classes, num_prompts).mean(dim=-1)
