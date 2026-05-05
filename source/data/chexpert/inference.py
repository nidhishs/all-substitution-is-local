"""DenseNet inference for CheXpert frontal images.

Heavy ML dependencies (torch, torchxrayvision, skimage) are imported lazily.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

import utils
from paths import dataset_prepared

from .labels import CONDITIONS, load_labels

logger = logging.getLogger("inference")

_PREPARED_ROOT = dataset_prepared("chexpert")


def run_inference(
    labels: pd.DataFrame,
    *,
    weights: str = "densenet121-res224-chex",
    output_path: Path,
    force: bool = False,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run DenseNet on all images; cache to output_path.

    Returns a DataFrame with competition-format columns:
    [Study, Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion].
    """
    import skimage.io
    import torch
    import torchxrayvision as xrv

    if output_path.exists() and not force:
        logger.info(f"Loading cached inference from {output_path}")
        return pd.read_csv(output_path)

    dev = utils.torch_device()
    logger.info(f"DenseNet device: {dev}")

    model = xrv.models.DenseNet(weights=weights)
    model.eval().to(dev)

    # torchxrayvision names the condition "Effusion"; competition CSVs use "Pleural Effusion"
    cond_idx = [
        model.pathologies.index("Effusion" if c == "Pleural Effusion" else c)
        for c in CONDITIONS
    ]

    def _preprocess(path: str) -> "torch.Tensor":
        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)
        img = xrv.datasets.XRayCenterCrop()(img[np.newaxis])
        img = xrv.datasets.XRayResizer(224)(img)
        return torch.from_numpy(img).unsqueeze(0)  # type: ignore[reportPrivateImportUsage]

    paths = labels["image_path"].tolist()
    raw = np.zeros((len(paths), len(CONDITIONS)), dtype=np.float32)

    logger.info(
        f"Running inference on {len(paths)} images (batch_size={batch_size}) ..."
    )
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            tensors = torch.cat([_preprocess(p) for p in batch], dim=0).to(dev)  # type: ignore[reportPrivateImportUsage]
            out = model(tensors).detach().cpu().numpy()
            raw[start : start + len(batch)] = out[:, cond_idx]
            if (start // batch_size) % 5 == 0:
                logger.info(f"  {start + len(batch)}/{len(paths)} images processed")

    inference_df = pd.DataFrame(raw, columns=list(CONDITIONS))
    inference_df.insert(0, "Study", labels["study_id"].values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inference_df.to_csv(output_path, index=False)
    logger.info(f"Saved inference cache: {output_path}")
    return inference_df


# fmt: off
@click.command()
@click.option( "--weights", default="densenet121-res224-chex", show_default=True, help="TorchXRayVision DenseNet weight string.")
@click.option("--batch-size", default=16, show_default=True, type=int, help="Inference batch size.")
@click.option("--force", is_flag=True, help="Re-run even if output CSV exists.")
# fmt: on
def main(weights: str, batch_size: int, force: bool) -> None:
    """Run DenseNet on all 500 images and write a competition-format predictions CSV."""
    model_name = weights.replace("/", "--")
    out_dir = _PREPARED_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.setup_logging(out_dir, "inference")
    utils.log_run_args(logger.info)
    labels = load_labels()
    run_inference(
        labels,
        weights=weights,
        batch_size=batch_size,
        output_path=out_dir / f"{model_name}.csv",
        force=force,
    )
