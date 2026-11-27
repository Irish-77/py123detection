"""Every config the package ships must compose, and mean what it says.

A broken config otherwise surfaces when someone starts a run, which on this kind of job means
after the queue, the data load and the first epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

pytest.importorskip("hydra", reason="the torch branch needs hydra")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

from hydra import compose, initialize_config_dir  # noqa: E402

from py123detection.core.taxonomy import TAXONOMIES  # noqa: E402

CONF = Path(__file__).resolve().parents[1] / "src" / "py123detection" / "torch" / "conf"


def group(name: str) -> List[str]:
    return sorted(path.stem for path in (CONF / name).glob("*.yaml"))


def composed(overrides: List[str]):
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize("model", group("model"))
@pytest.mark.parametrize("experiment", group("experiment"))
def test_every_model_composes_with_every_experiment(model: str, experiment: str) -> None:
    cfg = composed([f"model={model}", f"experiment={experiment}"])
    assert cfg.model._target_
    assert cfg.data.train and cfg.data.val


@pytest.mark.parametrize("taxonomy", group("taxonomy"))
def test_every_taxonomy_names_a_real_one(taxonomy: str) -> None:
    """A typo here would only surface once the data had loaded."""
    cfg = composed([f"taxonomy={taxonomy}"])
    assert cfg.taxonomy.name in TAXONOMIES


@pytest.mark.parametrize("taxonomy", group("taxonomy"))
def test_every_class_can_be_scored(taxonomy: str) -> None:
    """The metric needs a radius per class, and refuses to guess one."""
    cfg = composed([f"taxonomy={taxonomy}"])
    declared = cfg.taxonomy.class_range
    if declared is None:
        pytest.skip("uses the devkit's own nuScenes radii")

    for class_name in TAXONOMIES[cfg.taxonomy.name].class_names:
        assert class_name in declared, f"{class_name} has no detection range"
        assert declared[class_name] > 0


@pytest.mark.parametrize("trainer", group("trainer"))
def test_every_trainer_setting_is_usable(trainer: str) -> None:
    cfg = composed([f"trainer={trainer}"])
    assert cfg.trainer.max_epochs > 0
    assert cfg.trainer.devices
    assert cfg.trainer.gradient_clip_val > 0


@pytest.mark.parametrize("trainer", group("trainer"))
def test_extra_checkpoints_are_opt_in_and_inside_the_run(trainer: str, tmp_path) -> None:
    """Each named epoch costs a full checkpoint, so no shipped config asks for one.

    This also pins where they land: the run's own ``checkpoints/``, beside the best-three, rather
    than a directory of their own that a later tool would have to know about.
    """
    from py123detection.torch.callbacks import SaveEpochCheckpoints
    from py123detection.torch.train import build_callbacks

    cfg = composed([f"trainer={trainer}", f"paths.work_dir={tmp_path}"])
    assert not cfg.trainer.get("save_epoch_checkpoints")
    assert not [c for c in build_callbacks(cfg) if isinstance(c, SaveEpochCheckpoints)]

    cfg = composed([f"trainer={trainer}", f"paths.work_dir={tmp_path}", "+trainer.save_epoch_checkpoints=[6,12]"])
    extra = [c for c in build_callbacks(cfg) if isinstance(c, SaveEpochCheckpoints)]
    assert len(extra) == 1
    assert extra[0].dirpath == tmp_path / cfg.run.name / "checkpoints"


@pytest.mark.parametrize("model", group("model"))
def test_grid_mask_annealing_is_off_by_default(model: str) -> None:
    """Turning it on changes what a run learns, so no shipped config may do it quietly."""
    cfg = composed([f"model={model}"])
    assert cfg.model.grid_mask_annealing is False


def test_every_dataloader_setting_reaches_the_data_module() -> None:
    """A key the data module accepts but nothing passes is a knob that does not turn.

    ``prefetch_factor`` was exactly that: the data module took it, ``build_data`` never passed it,
    and it sat pinned at 4 with no way to change it.
    """
    import inspect

    from py123detection.torch.data import Py123DDataModule

    cfg = composed([])
    accepted = set(inspect.signature(Py123DDataModule).parameters)
    assert set(cfg.dataloader) <= accepted, (
        f"{set(cfg.dataloader) - accepted} is configured but not a Py123DDataModule parameter"
    )
    assert "prefetch_factor" in cfg.dataloader


@pytest.mark.parametrize("images", group("images"))
def test_image_sizes_suit_the_backbone(images: str) -> None:
    """A size the stride does not divide silently changes the feature grid the head assumes."""
    cfg = composed([f"images={images}"])
    height, width = cfg.images.size
    assert height % 16 == 0 and width % 16 == 0


class TestRunNaming:
    def test_each_combination_gets_its_own_directory(self) -> None:
        """Two runs sharing a directory overwrite each other's checkpoints."""
        names = {
            composed([f"model={model}", f"experiment={experiment}", f"taxonomy={taxonomy}"]).run.name
            for model in group("model")
            for experiment in group("experiment")
            for taxonomy in group("taxonomy")
        }
        expected = len(group("model")) * len(group("experiment")) * len(group("taxonomy"))
        assert len(names) == expected

    def test_the_name_says_what_was_run(self) -> None:
        name = composed(["model=bevformer", "experiment=nuscenes_av2", "taxonomy=mixed2_3cls"]).run.name
        assert name == "bevformer_nuscenes_av2_3cls"


class TestDefaults:
    def test_the_detection_range_reaches_the_head_and_the_filters(self) -> None:
        """Filtering to a different volume than the model can express loses boxes invisibly."""
        cfg = composed([])
        assert cfg.filters.pc_range == cfg.model.pc_range

    def test_the_positional_embedding_covers_more_than_the_detection_range(self) -> None:
        """An object at the edge of pc_range still needs an unsaturated embedding."""
        cfg = composed(["model=petr"])
        for detection, embedding in zip(cfg.model.pc_range[:3], cfg.model.position_range[:3]):
            assert embedding <= detection
        for detection, embedding in zip(cfg.model.pc_range[3:], cfg.model.position_range[3:]):
            assert embedding >= detection

    def test_a_mixed_experiment_uses_more_than_one_rig(self) -> None:
        cfg = composed(["experiment=carla_multirig"])
        assert len({entry.dataset for entry in cfg.data.train}) > 1
