"""Pretrained CIFAR-10 model inference on the 10 000 test images.

Heavy ML dependencies (torch, torchvision) are imported lazily inside run_inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

import utils
from paths import DATA_RAW, dataset_prepared

from .labels import CLASSES

logger = logging.getLogger("inference")

_PREPARED_ROOT = dataset_prepared("cifar10h")

# Supported pretrained weights from chenyaofo/pytorch-cifar-models (torch.hub).
SUPPORTED_WEIGHTS: tuple[str, ...] = (
    "cifar10_resnet20",
    "cifar10_resnet32",
    "cifar10_resnet44",
    "cifar10_resnet56",
    "cifar10_vgg16_bn",
    "cifar10_mobilenetv2_x1_0",
    "cifar10_repvgg_a2",
)

# Standard CIFAR-10 normalisation constants (used by chenyaofo/pytorch-cifar-models).
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def run_inference(
    *,
    weights: str,
    output_path: Path,
    force: bool = False,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Run a pretrained CIFAR-10 model on all 10 000 test images; cache to output_path.

    Returns a DataFrame with columns:
        image_id, airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    where the class columns are softmax probabilities (float32).
    """
    import pickle

    import torch
    import torchvision.transforms as T
    from PIL import Image

    if output_path.exists() and not force:
        logger.info(f"Loading cached inference from {output_path}")
        return pd.read_csv(output_path)

    dev = utils.torch_device()
    logger.info(f"Device: {dev}")

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        weights,
        pretrained=True,
        verbose=False,
        trust_repo=True,
    )
    model.eval().to(dev)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ]
    )

    # Read test_batch directly — avoids the torchvision MD5 integrity check, which fails
    # when the official Toronto server is unavailable and the batch was sourced elsewhere.
    test_batch_path = DATA_RAW / "cifar10h" / "cifar-10-batches-py" / "test_batch"
    with open(test_batch_path, "rb") as f:
        batch_data = pickle.load(f, encoding="bytes")
    images_raw = batch_data[b"data"]  # (10000, 3072) uint8, R/G/B planes concatenated

    n_images = len(images_raw)
    raw = np.zeros((n_images, len(CLASSES)), dtype=np.float32)

    logger.info(f"Running inference on {n_images} images (batch_size={batch_size}) ...")
    with torch.no_grad():
        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            imgs = []
            for flat in images_raw[start:end]:
                pil = Image.fromarray(flat.reshape(3, 32, 32).transpose(1, 2, 0))
                imgs.append(transform(pil))
            batch = torch.stack(imgs).to(dev)
            logits = model(batch).detach().cpu()
            probs = torch.softmax(logits, dim=1).numpy()
            raw[start:end] = probs
            if (start // batch_size) % 10 == 0:
                logger.info(f"  {end}/{n_images} images processed")

    df = pd.DataFrame(raw, columns=list(CLASSES))
    df.insert(0, "image_id", np.arange(n_images, dtype=np.int32))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved inference cache: {output_path}")
    return df


# fmt: off
@click.command()
@click.option("--weights", required=True, type=click.Choice(SUPPORTED_WEIGHTS), help="Pretrained model weights (chenyaofo/pytorch-cifar-models).")
@click.option("--batch-size", default=256, show_default=True, type=int, help="Inference batch size.")
@click.option("--force", is_flag=True, help="Re-run even if output CSV exists.")
# fmt: on
def main(weights: str, batch_size: int, force: bool) -> None:
    """Run a pretrained CIFAR-10 model and write a competition-format predictions CSV."""
    out_dir = _PREPARED_ROOT / weights
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.setup_logging(out_dir, "inference")
    utils.log_run_args(logger.info)
    run_inference(
        weights=weights,
        batch_size=batch_size,
        output_path=out_dir / f"{weights}.csv",
        force=force,
    )
