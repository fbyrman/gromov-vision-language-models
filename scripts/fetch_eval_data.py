"""
Fetch and verify the zero-shot evaluation datasets into a single root directory.

The evaluation configs read `data_dir="datasets/eval"` relative to the repository
root, so the root defaults there and may be pointed elsewhere with `EVAL_DATA_DIR`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
from functools import partial
from pathlib import Path

import gromovclip.data.evaluation as DE
from gromovclip.evaluation.catalog import DatasetCatalog

# Subclasses of TfdsWrapper hardcode the TFDS name they wrap rather than taking
# it as a keyword, so they cannot be resolved by inspecting the partial.
TFDS_SUBCLASSES: dict[type, str] = {DE.CLEVRCounts: "clevr"}

# Paths that must exist before a dataset is worth constructing. Constructing a
# torchvision dataset is what triggers its download, so presence is decided on
# disk first and the constructor only ever runs to count an existing copy.
PROBES: dict[str, list[str]] = {
    "aircraft": ["aircraft/fgvc-aircraft-2013b/data"],
    "cars": ["cars/stanford_cars"],
    "country211": ["country211/country211"],
    "cub2011": ["cub2011/CUB_200_2011/images"],
    "sst2": ["sst2/rendered-sst2"],
    "imagenet": ["imagenet/val"],
    "sun397": ["sun397/test"],
    "coco": ["coco/val2017", "coco/annotations"],
    "flickr30k": ["flickr30k/flickr30k_images", "flickr30k/dataset_flickr30k.json"],
}

# Datasets behind an account, a request form or a login wall. `download=True`
# cannot reach them, so they are reported rather than fetched.
MANUAL: dict[str, str] = {
    "imagenet": (
        "Register at image-net.org and download both ILSVRC2012_img_val.tar and\n"
        "  ILSVRC2012_devkit_t12.tar.gz into <root>/imagenet. Re-running this script\n"
        "  then sorts the flat archive into the per-wnid tree ImageFolder expects."
    ),
    "flickr30k": (
        "Request access at bryanplummer.com/Flickr30kEntities. Place the images at\n"
        "  <root>/flickr30k/flickr30k_images and Karpathy's dataset_flickr30k.json\n"
        "  at <root>/flickr30k/dataset_flickr30k.json."
    ),
    "resisc45": (
        "OneDrive only; see the TFDS resisc45 catalog page for the link. The rar\n"
        "  must be expanded elsewhere -- the builder wants the extracted directory,\n"
        "  not the archive -- leaving <root>/downloads/manual/NWPU-RESISC45 holding\n"
        "  one subdirectory per class."
    ),
    "cars": (
        "The Stanford server no longer serves the archives, so the catalog uses a\n"
        "  Kaggle mirror. Put a Kaggle API token at ~/.kaggle/kaggle.json, then this\n"
        "  script can fetch it."
    ),
}

# The TFDS eurosat builder points at madm.dfki.de, which now answers 403. Zenodo
# is the authors' own record; the tree under it is the same <class>/<class>_N.jpg
# the builder globs for, only the top directory is named differently.
EUROSAT_MIRROR = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
EUROSAT_SUBDIR = "EuroSAT_RGB"

KARPATHY_SPLITS = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)

# SUN397 is read from a parquet release on the Hugging Face Hub: the TFDS
# builder needs the 37 GB source tarball to produce about a gigabyte of records.
SUN397_REPO = "tanganke/sun397"
SUN397_REVISION = "9c4afd6174afa3974f209a0ef099dbbe244dfd27"

# The retrieval evaluator needs COCO images and captions side by side; torchvision
# has no downloader for the 2017 splits.
COCO_URLS = [
    ("http://images.cocodataset.org/zips/val2017.zip", "val2017"),
    (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations",
    ),
]


def default_root() -> Path:
    return Path(os.environ.get("EVAL_DATA_DIR", "datasets/eval"))


def default_downloads(root: Path) -> Path:
    """
    Where source archives are cached.

    Kept separable from the root because the archives are far larger than the
    datasets built from them, and only the built form has to sit on a filesystem
    the compute nodes can reach.
    """
    return Path(os.environ.get("EVAL_DOWNLOAD_DIR", root / "downloads"))


def tfds_name(name: str) -> str | None:
    ctor = DatasetCatalog.CONSTRUCTORS[name]
    if isinstance(ctor, partial):
        return ctor.keywords.get("name") if ctor.func is DE.TfdsWrapper else None
    return TFDS_SUBCLASSES.get(ctor)


def check(name: str, root: Path) -> tuple[bool, str]:
    """Report whether `name` is present and readable, without downloading it."""
    tname = tfds_name(name)
    if tname is not None:
        import tensorflow_datasets as tfds

        try:
            info = tfds.builder(tname, data_dir=str(root)).info
            splits = {k: v.num_examples for k, v in info.splits.items()}
            if not splits:
                return False, "no splits built"
            return True, ", ".join(f"{k}={v}" for k, v in splits.items())
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    missing = [p for p in PROBES.get(name, []) if not (root / p).exists()]
    if missing:
        return False, "missing " + ", ".join(missing)

    # Several torchvision constructors carry download=True, so counting a copy
    # that is present but incomplete would refetch it -- gigabytes, silently,
    # from something the caller asked only to inspect.
    ctor = DatasetCatalog.CONSTRUCTORS[name]
    offline = (
        partial(ctor.func, **{**ctor.keywords, "download": False})
        if isinstance(ctor, partial) and ctor.keywords.get("download")
        else None
    )
    if offline is not None:
        DatasetCatalog.CONSTRUCTORS[name] = offline
    try:
        dset = DatasetCatalog.build(name, str(root), "test")
        return True, f"{len(dset)} test samples"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        if offline is not None:
            DatasetCatalog.CONSTRUCTORS[name] = ctor


def prepare_imagenet(root: Path) -> None:
    """
    Sort the flat validation archive into one directory per wnid.

    The val tar holds 50,000 files named by index, and the index-to-wnid map only
    exists in the devkit, so both archives must be in place. Torchvision's
    parsers are used rather than a local reimplementation because the ILSVRC2012
    IDs in the ground truth are not in sorted-wnid order, which is the order
    `ImageFolder` -- and therefore `class_names.py` -- assumes.
    """
    from torchvision.datasets.imagenet import parse_devkit_archive, parse_val_archive

    target = root / "imagenet"
    for archive in ("ILSVRC2012_devkit_t12.tar.gz", "ILSVRC2012_img_val.tar"):
        if not (target / archive).is_file():
            raise FileNotFoundError(f"{target / archive} is required; see --check")

    parse_devkit_archive(str(target))
    parse_val_archive(str(target))


def fetch_karpathy_split(root: Path) -> None:
    import zipfile

    from torchvision.datasets.utils import download_url

    target = root / "flickr30k" / "dataset_flickr30k.json"
    if target.is_file():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    download_url(KARPATHY_SPLITS, str(target.parent))
    archive = target.parent / Path(KARPATHY_SPLITS).name
    with zipfile.ZipFile(archive) as z:
        z.extract("dataset_flickr30k.json", str(target.parent))
    archive.unlink()


def stage(src: Path, dst: Path, exclude: tuple[str, ...] = ()) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    args = ["rsync", "-a", *(f"--exclude={p}" for p in exclude)]
    subprocess.run([*args, f"{src}/", f"{dst}/"], check=True)


def fetch(
    name: str, root: Path, downloads: Path, work: Path | None = None
) -> None:
    if name == "imagenet":
        prepare_imagenet(root)
        return

    tname = tfds_name(name)
    if tname is not None:
        import tensorflow_datasets as tfds

        # A TFDS build reads every source image individually. On a network
        # filesystem that per-file latency dominates, so the build is done on
        # `work` -- node-local disk -- and only the finished shards are copied
        # back. The archive cache is staged once per run, not here.
        build_root = work if work is not None else root
        dl_dir = (work / "downloads") if work is not None else downloads

        builder = tfds.builder(tname, data_dir=str(build_root))
        if tname == "eurosat":
            builder.builder_config.download_url = EUROSAT_MIRROR
            builder.builder_config.subdir = EUROSAT_SUBDIR
        builder.download_and_prepare(
            download_dir=str(dl_dir),
            download_config=tfds.download.DownloadConfig(
                manual_dir=str(dl_dir / "manual")
            ),
        )
        if work is not None:
            stage(work / tname, root / tname)
            # Freed immediately so node-local usage stays bounded by the archive
            # cache plus one dataset, rather than the whole suite.
            shutil.rmtree(work / tname, ignore_errors=True)
        return

    if name == "flickr30k":
        # Only the images are behind the request form; Karpathy's split file is
        # public, so it is fetched here to narrow what has to be done by hand.
        fetch_karpathy_split(root)
        raise FileNotFoundError(
            f"{root / 'flickr30k' / 'flickr30k_images'} must be placed by hand"
        )

    if name == "sun397":
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            SUN397_REPO,
            repo_type="dataset",
            revision=SUN397_REVISION,
            allow_patterns="data/test-*.parquet",
        )
        target = root / "sun397" / "test"
        target.mkdir(parents=True, exist_ok=True)
        for shard in sorted(Path(snapshot, "data").glob("test-*.parquet")):
            shutil.copy(shard, target / shard.name)
        return

    if name == "coco":
        from torchvision.datasets.utils import download_and_extract_archive

        for url, member in COCO_URLS:
            if (root / "coco" / member).is_dir():
                continue
            download_and_extract_archive(url, download_root=str(root / "coco"))
        return

    # Torchvision constructors download on instantiation; the evaluators only
    # ever read the test split.
    DatasetCatalog.build(name, str(root), "test")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(DatasetCatalog.CONSTRUCTORS),
        help="Defaults to every dataset in the catalog.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Archive cache; defaults to <root>/downloads or EVAL_DOWNLOAD_DIR.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=os.environ.get("EVAL_WORK_DIR"),
        help="Node-local directory to build TFDS datasets in before copying to --root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what is present and exit without downloading.",
    )
    args = parser.parse_args(argv)

    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    downloads = args.download_dir or default_downloads(root)
    work = None if args.check else args.work_dir
    if work is not None and downloads.is_dir():
        # One pass up front: the cache is tens of GB and every TFDS build would
        # otherwise re-walk it. Downloads made on `work` are copied back at the
        # end so a later run does not refetch them. Already-extracted trees are
        # left behind -- they are the many-small-files case a network filesystem
        # is worst at, and re-extracting the archives locally is far quicker.
        print(f"STAGE   downloads -> {work}", flush=True)
        stage(downloads, work / "downloads", exclude=("extracted/",))

    failed, manual = [], []
    for name in args.datasets:
        if name not in DatasetCatalog.CONSTRUCTORS:
            print(f"SKIP    {name:12s} not in catalog")
            continue

        ok, detail = check(name, root)
        if ok:
            print(f"OK      {name:12s} {detail}")
            continue
        if args.check:
            print(f"MISSING {name:12s} {detail}")
            (manual if name in MANUAL else failed).append(name)
            continue

        print(f"FETCH   {name:12s} ({detail})", flush=True)
        try:
            fetch(name, root, downloads, work)
        except Exception:
            traceback.print_exc()
            (manual if name in MANUAL else failed).append(name)
            print(f"FAIL    {name:12s}")
            continue

        ok, detail = check(name, root)
        print(f"{'OK     ' if ok else 'FAIL   '} {name:12s} {detail}")
        if not ok:
            (manual if name in MANUAL else failed).append(name)

    if work is not None and (work / "downloads").is_dir():
        stage(work / "downloads", downloads, exclude=("extracted/",))

    if manual:
        print("\nThese need manual acquisition:")
        for name in manual:
            print(f"\n{name}\n  {MANUAL[name]}")

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
