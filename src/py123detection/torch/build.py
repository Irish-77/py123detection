"""Turning a Hydra config into the objects a training run needs.

Kept apart from the entry points so that training, evaluation and a notebook all build their data
the same way, and so the mapping from config to object is one readable file rather than something
spread through two scripts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from py123detection.core.sources import Source
from py123detection.core.taxonomy import get_taxonomy
from py123detection.torch.data import (
    Augmentation,
    BoxFilters,
    ImagePipeline,
    Py123DDetectionDataset,
)
from py123detection.torch.data.rigs import get_rig

logger = logging.getLogger(__name__)


def _plain(node: Any) -> Any:
    """A config node as ordinary Python, so it can be splatted into a constructor."""
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def build_images(cfg: DictConfig) -> ImagePipeline:
    """The network input, shared by training and validation.

    Only the size and the resizing mode live here. Everything random is in ``augment``, so the
    validation data is simply the same pipeline without an augmentation.
    """
    options: Dict[str, Any] = _plain(cfg.images)
    return ImagePipeline(size=tuple(options.pop("size")), **options)


def build_filters(cfg: DictConfig) -> BoxFilters:
    return BoxFilters(**_plain(cfg.filters))


def build_augmentation(cfg: DictConfig) -> Optional[Augmentation]:
    if cfg.get("augment") is None:
        return None
    options = _plain(cfg.augment)
    return Augmentation(
        zoom_range=tuple(options["zoom_range"]),
        flip=bool(options["flip"]),
        rotation_range=tuple(options["rotation_range"]),
        scale_range=tuple(options["scale_range"]),
    )


def build_datasets(cfg: DictConfig, stage: str) -> List[Py123DDetectionDataset]:
    """One dataset per entry under ``data.<stage>``.

    Training entries get augmentation, validation entries never do. Everything else — the
    taxonomy, the filters, the image size — is shared, which is what makes a mixed-dataset run a
    fair comparison rather than several differently prepared ones.
    """
    entries = cfg.data.get(stage)
    if not entries:
        raise ValueError(f"Config has no data.{stage} entries; pick an experiment=<name>.")

    training = stage == "train"
    taxonomy = get_taxonomy(cfg.taxonomy.name)
    images = build_images(cfg)
    augmentation = build_augmentation(cfg) if training else None
    filters = build_filters(cfg)

    datasets = []
    for entry in entries:
        spec = _plain(entry)
        name = spec.pop("dataset")
        if spec.get("sample_rate_hz") is not None and spec.get("frame_stride") is not None:
            raise ValueError(
                f"data.{stage} entry '{name}' sets both sample_rate_hz and frame_stride; they are two "
                f"ways of saying the same thing, so choose one."
            )
        source = Source(
            data_root=cfg.paths.get("data_root"),
            splits=spec.pop("splits", None),
            sample_rate_hz=spec.pop("sample_rate_hz", None),
            frame_stride=spec.pop("frame_stride", None),
            max_logs=spec.pop("max_logs", None),
            name=name,
        )
        datasets.append(
            Py123DDetectionDataset(
                source=source,
                taxonomy=taxonomy,
                rig=get_rig(name),
                images=images,
                augment=augmentation,
                filters=filters,
                name=name,
                seed=cfg.run.seed,
                **spec,
            )
        )
        logger.info("%s: %s (%d frames)", stage, name, len(datasets[-1]))

    return datasets


def build_model(cfg: DictConfig, **overrides: Any):
    """The detector named by ``cfg.model``.

    ``name`` is metadata — it labels the run directory — so it is dropped before the rest of the
    node is handed to the constructor.
    """
    from hydra.utils import instantiate

    options = OmegaConf.masked_copy(
        cfg.model, [key for key in cfg.model if key not in ("name",)]
    )
    return instantiate(options, **overrides, _convert_="all")
