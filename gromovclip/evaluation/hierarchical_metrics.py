from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import nltk
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from gromovclip.evaluation.catalog import DatasetCatalog
from gromovclip.evaluation.class_names import CLASS_NAMES
from gromovclip.evaluation.classification import (
    ZeroShotClassificationEvaluator,
    _encode_dataset,
)
from gromovclip.models import MERU, CLIPBaseline, HyCoCLIP, PHyCLIP
from gromovclip.tokenizer import Tokenizer
from gromovclip.utils.evaluation import (
    compute_similarity_scores,
    compute_tangent_similarity_scores,
    process_text_features,
    process_text_features_tangent,
    score_level_classifier,
    score_level_classifier_tangent,
    score_level_scores,
    score_level_scores_tangent,
    to_tangent_features,
)

nltk.download("wordnet")
from nltk.corpus import wordnet as wn  # noqa: E402

IMAGENET_SYNSET_ORDER = "assets/imagenet_synset/all_synsets.pkl"
IMAGENET_WORDNET_GRAPH = "assets/imagenet_synset/imagenet_isa.txt"
ANCESTOR_INDICES = "assets/imagenet_synset/all_ancestors_indices.pkl"


def def_value():
    return "Not Present"


def create_nx_graph_from_edges(edge_file):
    """
    Create a NetworkX DiGraph object from a file containing edges of the graph.
    """
    synset_map = {}

    for s in wn.all_synsets():
        synset_map[s.offset()] = s.name().split(".")[0]

    graph = nx.DiGraph()
    edge_dict = defaultdict(def_value)
    synset_to_label = defaultdict(def_value)

    with open(edge_file, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            edge_dict.setdefault(row[0], []).append(row[1])  # type: ignore
            if row[0] not in list(graph.nodes):
                graph.add_node(row[0])
                synset_to_label[row[0]] = synset_map[int(row[0][1:])]
            if row[1] not in list(graph.nodes):
                graph.add_node(row[1])
                synset_to_label[row[1]] = synset_map[int(row[1][1:])]

    for parent, children in edge_dict.items():
        for child in children:
            graph.add_edge(parent, child)

    return graph


def hierarchical_based_metrics(
    predicted_labels, true_labels, ancestor_indices, graph, synsets_ordering
):
    """
    Calculate Hierarchical metrics
    """
    undirected_graph = graph.to_undirected()
    tree_induced_error = 0
    least_common_ancestor = 0
    jaccard = 0
    hierarchical_precision = 0
    hierarchical_recall = 0

    for _b in range(predicted_labels.size(0)):
        pred_label = predicted_labels[_b]
        pred_synset = synsets_ordering[pred_label]
        true_label = true_labels[_b]
        true_synset = synsets_ordering[true_label]
        pred_ancestors = ancestor_indices[pred_label]
        true_ancestors = ancestor_indices[true_label]
        pred_ancestors = set(pred_ancestors)
        true_ancestors = set(true_ancestors)

        intersection = pred_ancestors.intersection(true_ancestors)
        union = pred_ancestors.union(true_ancestors)

        tree_induced_error += nx.shortest_path_length(
            undirected_graph, source=pred_synset, target=true_synset
        )
        least_common_ancestor += (
            len(pred_ancestors) - len(intersection) + 1
        )  # +1 for the actual class label
        jaccard += len(intersection) / len(union)
        hierarchical_precision += len(intersection) / len(pred_ancestors)
        hierarchical_recall += len(intersection) / len(true_ancestors)

    return (
        tree_induced_error,
        least_common_ancestor,
        jaccard,
        hierarchical_precision,
        hierarchical_recall,
        predicted_labels.size(0),
    )


class HierarchicalMetricsEvaluator(ZeroShotClassificationEvaluator):
    """
    Evaluate model performance on Hierarchical metrics:
    1. Tree Induced Error (TIE)
    2. Least Common Ancestor (LCA)
    3. Jaccard Similarity
    4. Hierarchical Precision
    5. Hierarchical Recall
    """

    def __init__(
        self,
        datasets_and_prompts: dict[str, list[str]],
        data_dir: str | Path,
        image_size: int = 224,
        ensembling: str = "feature",
        exp_map: bool = True,
    ):
        """
        Args:
            datasets_and_prompts: Dictionary mapping between dataset name and
                a list of prompt templates to fill using its class names. Add
                a single `{}` in prompt to fill with class name. Datasets
                should be among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
            ensembling: How the prompts of a class are combined. `"feature"`
                averages them in the tangent space and lifts the mean, so a
                class is one point. `"score"` lifts every prompt on its own
                and scores a class by the mean distance of its prompts to the
                image; it needs a model whose scores are distances (a `MERU`
                subclass).
            exp_map: If False, read a hyperbolic model without the exponential
                map: encoder outputs are scaled by the learned alpha and
                compared by Euclidean distance. Models that are not `MERU`
                subclasses have no exponential map to remove and keep their
                default scoring.
        """
        super().__init__(datasets_and_prompts, data_dir, image_size, ensembling, exp_map)

    @torch.inference_mode()
    def __call__(
        self, model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline
    ) -> dict[str, float]:
        model = model.eval()
        tokenizer = Tokenizer()

        _score = self._ensembling == "score"
        _tangent = not self._exp_map and isinstance(model, MERU)
        if _score and not isinstance(model, MERU):
            raise TypeError(
                f"ensembling='score' needs a MERU subclass, got {type(model).__name__}"
            )

        imagenet_synset_order = pickle.load(open(IMAGENET_SYNSET_ORDER, "rb"))
        ancestor_indices = pickle.load(open(ANCESTOR_INDICES, "rb"))
        graph = create_nx_graph_from_edges(IMAGENET_WORDNET_GRAPH)

        results_dict = {}

        for dname, prompts in self._datasets_and_prompts.items():
            class_names = CLASS_NAMES[dname]

            # Collect text features of each class.
            all_class_feats: list[torch.Tensor] = []

            for i, name in enumerate(class_names):
                class_prompts = [_pt.format(name) for _pt in prompts]

                class_prompt_tokens = tokenizer(class_prompts)
                class_feats = model.encode_text(class_prompt_tokens, project=False)  # type: ignore
                if _score and _tangent:
                    class_feats = score_level_classifier_tangent(model, class_feats)
                elif _tangent:
                    class_feats = process_text_features_tangent(model, class_feats)
                elif _score:
                    class_feats = score_level_classifier(model, class_feats)
                else:
                    class_feats = process_text_features(model, class_feats)

                all_class_feats.append(class_feats)

            # shape: (num_classes, embed_dim), or (num_classes, num_prompts,
            # embed_dim) when prompts are ensembled over scores, not features.
            classifier = torch.stack(all_class_feats, dim=0)

            # Extract image features and labels from the test split of required dataset.
            loader = DataLoader(
                DatasetCatalog.build(
                    dname, self._data_dir, "test", self._image_transform
                ),
                batch_size=512,  # Smaller batch size to prevent memory issues
                num_workers=8,  # Single process to avoid deadlocks
                pin_memory=False,  # Disable pin_memory to avoid GPU memory conflicts
                persistent_workers=False,  # Not needed with num_workers=1
                drop_last=False,  # Process all data
            )

            image_feats, labels = _encode_dataset(loader, model, project=not _tangent)

            # Features returned by this function will be on CPU, move to device:
            image_feats = image_feats.to(model.device)
            if _tangent:
                image_feats = to_tangent_features(model, image_feats, "image")

            # Measure model performance according to accuracy metric of the dataset.
            total_tie = 0
            total_lca = 0
            total_jaccard = 0
            total_hier_precision = 0
            total_hier_recall = 0
            total_val_size = 0

            # Evaluate in small batches of 256 instances.
            for _feats, _labels in tqdm(
                zip(image_feats.split(256), labels.split(256)),
                desc="Calculating scores",
            ):
                # Compute pairwise similarity depending on model type:
                if _score and _tangent:
                    scores = score_level_scores_tangent(model, _feats, classifier)
                elif _tangent:
                    scores = compute_tangent_similarity_scores(model, _feats, classifier)
                elif _score:
                    scores = score_level_scores(model, _feats, classifier)
                else:
                    scores = compute_similarity_scores(model, _feats, classifier)

                predicted_labels = scores.argmax(dim=-1)
                (
                    tree_induced_error,
                    least_common_ancestor,
                    jaccard,
                    hierarchical_precision,
                    hierarchical_recall,
                    b_size,
                ) = hierarchical_based_metrics(
                    predicted_labels,
                    _labels,
                    ancestor_indices,
                    graph,
                    imagenet_synset_order,
                )
                total_tie += tree_induced_error
                total_lca += least_common_ancestor
                total_jaccard += jaccard
                total_hier_precision += hierarchical_precision
                total_hier_recall += hierarchical_recall

                total_val_size += b_size

            avg_tie = total_tie / total_val_size
            avg_lca = total_lca / total_val_size
            avg_jaccard = total_jaccard / total_val_size
            avg_hier_precision = total_hier_precision / total_val_size
            avg_hier_recall = total_hier_recall / total_val_size

            logger.info(
                f"Hierarchical evaluation: {dname}, {len(image_feats)} images, "
                f"{len(class_names)} classes \n[tie score: {avg_tie:.3f}, lca score: {avg_lca:.3f}, "
                f"jaccard similarity: {avg_jaccard:.3f}, hierarchical precision: {avg_hier_precision:.3f}, "
                f"hierarchical recall: {avg_hier_recall:.3f}] "
            )

            results_dict[dname] = {
                "tie": avg_tie,
                "lca": avg_lca,
                "jaccard": avg_jaccard,
                "hierarchical_precision": avg_hier_precision,
                "hierarchical_recall": avg_hier_recall,
            }

        return results_dict
