# GromovCLIP

Code for *Gromov Vision-Language Models*. GromovCLIP trains a hyperbolic
vision-language model with the Gromov loss in place of entailment cones. This
repository contains GromovCLIP and the three baselines of the paper (HyCoCLIP,
ARGENT, PHyCLIP), all trained with grounded box supervision on CC3M and on GRIT
at ViT-S/16 and ViT-B/16.

Upon publication, we will share a Google Drive link to our GRIT shards, with
the data loader that reads them, and the model checkpoints on Hugging Face.

## The loss

For a parent $\boldsymbol{x}$ and a child $\boldsymbol{y}$ on the hyperboloid,
the Gromov ratio divides the Gromov product at the origin by the parent's
distance from the origin:

$$ r(\boldsymbol{x}, \boldsymbol{y}) = \frac{(\boldsymbol{x}, \boldsymbol{y})_{\boldsymbol{0}}}{d(\boldsymbol{x}, \boldsymbol{0})} \in [0, 1], \qquad \ell_{\text{Gromov}}(\boldsymbol{x}, \boldsymbol{y}) = \big(1 - r(\boldsymbol{x}, \boldsymbol{y})\big)^{k}. $$

The ratio is 1 when the parent lies on the geodesic from the origin to the
child, and falls to 0 as the two branch at the origin.

GromovCLIP keeps HyCoCLIP's contrastive loss and applies the Gromov loss to the
same four relations:

| relation | type | exponent |
| --- | --- | --- |
| text → image | inter-modal | `k_inter = 2` |
| box text → box image | inter-modal | `k_inter = 2` |
| box image → image | intra-modal | `k_intra = 3` |
| box text → text | intra-modal | `k_intra = 3` |

The total loss is `L_cont + gromov_weight * (sum of the four Gromov losses)`,
with `gromov_weight = 0.5`. The code is `gromov_loss` and `GromovCLIP` in
`gromovclip/models.py`, and `gromov_ratio` in `gromovclip/lorentz.py`.

## Layout

```
gromovclip/                  model code: Lorentz geometry, encoders, models, losses, evaluators
  models.py                  GromovCLIP, HyCoCLIP, ARGENT, PHyCLIP
  lorentz.py                 hyperboloid operations, including the Gromov ratio
  euclidean.py               Euclidean distances, for the read-out without the exponential map
configs/
  train_<method>_vit_{s,b}.py    one training config per method and encoder
  train_<method>_grit_vit_{s,b}.py   the same on GRIT, for 200K iterations
  cc3m_dataset.py                the CC3M training data, shared by every CC3M config
  grit_dataset.py                the GRIT training data, shared by every GRIT config
  eval_*.py                      zero-shot classification, retrieval, hierarchical metrics
scripts/
  train.py, evaluate.py          training and evaluation entry points
  extract_gligen_cc3m.py         GLIGEN CC3M TSVs -> webdataset shards
  build_manifest.py              per-shard sidecars -> manifest.json
  sanity_check_cc3m.py           structural and visual check of one shard
  fetch_eval_data.py             fetches and verifies the evaluation datasets
  export_geometry.py             data behind the geometry figure
  embed_texts.py                 text embeddings for the qualitative figures
  project_text_trees.py          HoroPCA projection of the text trees
figures/                     figure scripts (Figures 1, 3, 4, 5), writing to outputs/figures/
assets/                      ImageNet/WordNet hierarchy, texts for the qualitative figures
results/                     evaluation results behind the tables, data behind the figures
```

`train_<method>_vit_b.py` holds the full config. The `_vit_s` variant overrides
only `model.visual.arch`.

| config | model |
| --- | --- |
| `train_gromovclip_vit_{s,b}.py` | GromovCLIP |
| `train_hycoclip_vit_{s,b}.py` | HyCoCLIP |
| `train_argent_vit_{s,b}.py` | ARGENT |
| `train_phyclip_vit_{s,b}.py` | PHyCLIP |

Each hyperbolic model also has a GRIT config, `train_<method>_grit_vit_{s,b}.py`
for `gromovclip`, `hycoclip`, `argent` and `phyclip`. It imports the CC3M
config of the same model and changes only the dataset and the number of
iterations.

## Environment

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements-frozen.txt \
  --find-links https://download.pytorch.org/whl/cu124/torch \
  --find-links https://download.pytorch.org/whl/cu124/torchvision
uv pip install --python .venv/bin/python -e . --no-deps
uv pip install --python .venv/bin/python tensorflow-cpu==2.21.0
```

`requirements-frozen.txt` pins the training environment, CUDA 12.4 wheels
included. Evaluation also needs TensorFlow, which `tensorflow-datasets` uses to
read most of the classification datasets. The last line adds it without
changing any pinned version. `pyproject.toml` holds looser ranges for other
CUDA versions, with TensorFlow under the `eval` extra.

Every command below runs from the repository root.

## Data

Training uses CC3M images paired with GLIGEN's box annotations. Samples without
boxes are dropped at load time.

```bash
python scripts/extract_gligen_cc3m.py --out-dir datasets/cc3m_grounded
python scripts/build_manifest.py --data-dir datasets/cc3m_grounded --verify
python scripts/sanity_check_cc3m.py --shard train-0000
```

The source is [`gligen/cc3m_tsv`](https://huggingface.co/datasets/gligen/cc3m_tsv)
at revision `3f8d464e87669c5350787bdbc547f93dbb62357a` (483 TSV shards, about
518 GB). The TSVs carry the JPEGs, so no separate image download is needed and
no samples are lost to dead links. Shards are streamed over HTTP and not staged
on disk. A shard counts as done once its `.stats.json` sidecar exists, so an
interrupted run resumes when the command is repeated. `--shard-start` and
`--num-shards` split the shards over parallel processes.

The result is 483 shards with 3,016,640 samples (125 GB). Each sample holds a
`.jpg` (shortest side 256), a `.txt` caption and a `.json` with the boxes,
phrases and token spans.

## Training

```bash
export CC3M_GROUNDED_DIR=$PWD/datasets/cc3m_grounded
python scripts/train.py --config configs/train_gromovclip_vit_b.py --num-gpus 8 --resume
```

All runs are 100,000 iterations at a global batch size of 512, so
`--num-gpus` changes throughput, not the optimization. Any config field can be
overridden on the command line, e.g. `model.k_inter=1.0`.

On GRIT, runs are 200,000 iterations at the same batch size:

```bash
export GRIT_DIR=/path/to/grit
python scripts/train.py --config configs/train_gromovclip_grit_vit_b.py --num-gpus 8 --resume
```

The GRIT configs need the GRIT shards and their data loader
(`gromovclip/data/grit_grounded.py`), which we will share upon publication.

Each run writes to `train_results/<parameters>/`, named after the config's
parameters. `--resume` does nothing on a fresh run, so an interrupted run
resumes when the same command is repeated.

## Evaluation

```bash
python scripts/fetch_eval_data.py
```

This fetches the evaluation datasets into `datasets/eval` (or `EVAL_DATA_DIR`)
and verifies each one. Four cannot be fetched unattended, and the script prints
what to do for each: ImageNet needs an image-net.org account, RESISC45 is a
manual OneDrive download, Stanford Cars comes from a Kaggle mirror and needs a
token at `~/.kaggle/kaggle.json`, and the Flickr30K images are behind a request
form. `--check` reports what is present without downloading.

The paper reports score-level prompt ensembling: each prompt is lifted to the
manifold separately, and a class scores as the mean distance over its prompts.

```bash
python scripts/evaluate.py \
    --checkpoint-path train_results/<run>/checkpoint_final.pth \
    --train-config configs/train_gromovclip_vit_b.py \
    --ensembling score --results-file evaluate_score.json
```

Without `--config`, this runs classification, retrieval and hierarchical
metrics in turn. `--config configs/eval_<task>.py` runs one task. Results are
appended to `--results-file` next to the checkpoint.

- `--ensembling feature` averages the prompts in the tangent space and lifts the mean.
- `--no-exp-map` reads a hyperbolic model on Euclidean distances between the
  scaled encoder outputs, without the exponential map (the read-out ablation).

Give each read-out its own `--results-file`.

## Results

`results/benchmarks/<data>_<model>_vit_<s|b>.json` holds the evaluation results
behind the paper's tables: one file per pre-training set (CC3M, GRIT), model
and encoder.

| key | contents |
| --- | --- |
| `classification.score`, `classification.feature` | zero-shot accuracy (%) on 16 datasets, score- and feature-level prompt ensembling |
| `retrieval` | recall (%) at 5 and 10 on COCO and Flickr30K |
| `hierarchical.score`, `hierarchical.feature` | TIE, LCA, Jaccard, hierarchical precision and recall on ImageNet |
| `without_exp_map` | CC3M only: the same, read out without the exponential map (score level) |

| paper table | files and keys |
| --- | --- |
| Table 1 and its ViT-S/16 counterpart | all files: `classification.score`, `retrieval`, `hierarchical.score` |
| Read-outs (appendix) | all files: `classification.*`, `hierarchical.*` |
| Exponential map at read-out | `cc3m_*`: top level and `without_exp_map` |

Values are unrounded; the tables round half up.

## Figures

Figures 1 and 3 need neither data nor a GPU:

```bash
python figures/cone_disk.py        # Figure 1, entailment loss
python figures/gromov_disk.py      # Figure 1, Gromov loss
python figures/grad_flow_tikz.py   # Figure 3, as TikZ plus three magnitude images
```

`grad_flow_tikz.py` writes `outputs/figures/tikz/grad_flow.tex`, which uses the
paper's macros and colours. `grad_flow.py` draws the same gradient fields as a
standalone PDF.

```bash
python scripts/export_geometry.py --arch b --out results/geometry_vit_b.json
python figures/geometry.py
```

The geometry figure shows the learned curvature, the distance from the origin
and the distance distortion. `export_geometry.py` reads the trained runs from
`train_results/`. The exported data is included in `results/`, so the figure
scripts run without checkpoints.

The text-tree figures embed the texts in `assets/qualitative/` with each model,
project them with [HoroPCA](https://github.com/HazyResearch/HoroPCA), and draw
them:

```bash
python scripts/embed_texts.py --train-config configs/train_gromovclip_vit_b.py \
    --checkpoint-path train_results/<run>/checkpoint_final.pth \
    --texts assets/qualitative/text_trees.json assets/qualitative/background_texts.json \
    --out outputs/qualitative/gromovclip.pt
python scripts/project_text_trees.py --horopca-dir /path/to/HoroPCA \
    --embeddings gromovclip=outputs/qualitative/gromovclip.pt ... \
    --out results/text_trees.json
python figures/text_tree_plane.py boat --no-key
python figures/text_trees_tikz.py boat
python figures/text_tree_plane_all.py
```

## Credit

The code is built on the public code of
[MERU](https://github.com/facebookresearch/meru),
[HyCoCLIP](https://github.com/PalAvik/hycoclip) and
[PHyCLIP](https://github.com/tksmatsubara/PHyCLIP), and keeps their structure
(lazy configs, the webdataset training loop, the evaluation catalog). Files
adapted from these repositories say so in their header.

## License

CC BY-NC 4.0, as MERU. See `LICENSE`.
