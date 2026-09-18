"""Export a 123D nuScenes conversion to mmdetection3d info pickles.

Run it against a 123D data root that holds a nuScenes conversion::

    export PY123D_DATA_ROOT=/data/py123d
    export NUSCENES_DATA_ROOT=/data/nuscenes
    python examples/export_nuscenes.py --out-dir /data/nuscenes

Passing ``--nuscenes-root`` restores native nuScenes ``sample_token`` values, which the official
evaluation needs; without it the exported ``token`` is 123D's deterministic frame UUID.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from py123detection import Source
from py123detection.mmcv_export import ExportConfig, NuScenesTokenResolver, TokenResolver, export_to_mmdet3d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=None, help="123D root (default: $PY123D_DATA_ROOT).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, default=None, help="Restores native sample tokens.")
    parser.add_argument("--sample-rate-hz", type=float, default=None)
    parser.add_argument("--prefix", default="py123d")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # `version` ends up as metadata['version']; mmdet3d looks it up in its nuScenes eval_set_map,
    # so keep it a real nuScenes version string if you plan to evaluate.
    config = ExportConfig(
        token_resolver=NuScenesTokenResolver(args.nuscenes_root) if args.nuscenes_root else TokenResolver(),
    )

    for split, suffix in (("nuscenes-mini_train", "train"), ("nuscenes-mini_val", "val")):
        report = export_to_mmdet3d(
            Source(data_root=args.data_root, splits=[split], sample_rate_hz=args.sample_rate_hz),
            output_path=args.out_dir / f"{args.prefix}_infos_{suffix}.pkl",
            config=config,
            version="v1.0-mini",
        )
        print(report.summary())
        print()

    print("class_names for your mmdet3d config:")
    print(f"  {list(config.taxonomy.class_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
