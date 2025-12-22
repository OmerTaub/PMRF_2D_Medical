import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so that `import data` works when this
# script is executed from the `scripts` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import VanillaSliceData, create_mask_for_mask_type


def _tensor_stats(name: str, t: torch.Tensor | None, max_print: int = 8) -> None:
    if t is None:
        print(f"{name}: None")
        return
    flat = t.detach().reshape(-1)
    n = int(flat.numel())
    head = flat[: min(n, max_print)].cpu()
    print(
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={float(flat.min().cpu())} max={float(flat.max().cpu())} "
        f"head={head.tolist()}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Quick sanity-check for VanillaSliceData (raw k-space only)."
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Directory containing fastMRI .h5 files (e.g. singlecoil_train).",
    )
    p.add_argument(
        "--challenge",
        type=str,
        choices=["singlecoil", "multicoil"],
        default="singlecoil",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_items", type=int, default=3, help="How many items to print.")
    p.add_argument("--sample_rate", type=float, default=1.0)

    # Optional: generate a synthetic mask (instead of using file mask).
    p.add_argument("--use_mask_func", action="store_true")
    p.add_argument(
        "--mask_type",
        type=str,
        default="random",
        choices=["random", "equispaced"],
    )
    p.add_argument("--use_seed", action="store_true")
    p.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
    )
    p.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data_root is not a directory: {data_root}")

    mask_func = None
    if args.use_mask_func:
        mask_func = create_mask_for_mask_type(
            args.mask_type, args.center_fractions, args.accelerations
        )

    ds = VanillaSliceData(
        root=data_root,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        mask_func=mask_func,
        use_seed=args.use_seed,
    )
    print(f"VanillaSliceData: {len(ds)} slices from {data_root}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: batch,  # keep dicts/lists intact (targets may be None)
    )

    shown = 0
    for batch in loader:
        for item in batch:
            print("-" * 80)
            print(f"fname={item['fname']} slice={item['slice']}")
            _tensor_stats("kspace_full", item["kspace_full"])
            _tensor_stats("kspace_sub", item["kspace_sub"])
            _tensor_stats("mask", item.get("mask", None))
            _tensor_stats("target", item.get("target", None))
            shown += 1
            if shown >= args.num_items:
                return


if __name__ == "__main__":
    main()


