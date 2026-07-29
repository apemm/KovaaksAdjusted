"""Training loop for the FlickEncoder.

Seeded and deterministic (fixed permutation generator, cudnn determinism
flags), AdamW with a cosine schedule, early stopping on validation loss
with best-weights restore. Checkpoints land at
``<out_dir>/flick_encoder.pt`` (state dict + model config + target
normalization) with a sibling ``flick_encoder.json`` carrying human-
readable metadata (dataset size, losses, device, date).

Targets are standardized per head with train-split mean/std; those
statistics ride along in the checkpoint so inference de-normalizes with
the exact values training used. All losses reported here are MSE in
standardized units.

This module's header is torch-free (only ``model.py`` needs torch at
import time): ``train()`` returns ``None`` when torch is missing and
raises ``RuntimeError`` with a human-readable message when there is not
enough flick data to fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .dataset import N_SAMPLES, RATE, TRAIN_HEADS, build_dataset

CHECKPOINT_NAME = "flick_encoder.pt"
METADATA_NAME = "flick_encoder.json"

#: Below this many flicks a fit is noise; the CLI surfaces the message.
MIN_FLICKS = 64


@dataclass
class TrainResult:
    checkpoint: Path
    metadata: Path
    device: str
    params: int
    n_traces: int
    train_size: int
    val_size: int
    epochs_run: int
    best_epoch: int
    train_loss: float                       # final epoch, standardized MSE
    val_loss: float                         # best epoch, standardized MSE
    val_loss_per_head: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)


def train(
    traces_root: Path | str,
    out_dir: Path | str,
    *,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    patience: int = 10,
    device: str | None = None,
    rate: float = RATE,
    n_samples: int = N_SAMPLES,
    min_flicks: int = MIN_FLICKS,
) -> TrainResult | None:
    """Fit the FlickEncoder on every trace under ``traces_root``.

    Returns ``None`` when torch is not installed. Raises ``RuntimeError``
    when the trace library yields fewer than ``min_flicks`` usable flicks
    (or an empty split).
    """
    try:
        import torch
    except Exception:
        return None
    from .model import FlickEncoder, count_parameters

    ds = build_dataset(traces_root, rate=rate, n_samples=n_samples)
    assert ds is not None  # torch imported above
    n_train, n_val = int(ds.x_train.shape[0]), int(ds.x_val.shape[0])
    if ds.n_flicks < min_flicks or n_train == 0 or n_val == 0:
        raise RuntimeError(
            f"not enough flick data to train: {ds.n_flicks} flicks from "
            f"{ds.n_traces} traces under {traces_root} (need >= {min_flicks} "
            "with a non-empty train and val split). Record more sessions "
            'with `kovadapt watch "<scenario>"` first.'
        )

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Standardize targets on the train split; stats ship in the checkpoint.
    y_train = ds.y_train[:, : len(TRAIN_HEADS)]
    y_val = ds.y_val[:, : len(TRAIN_HEADS)]
    mean = y_train.mean(dim=0)
    std = y_train.std(dim=0).clamp_min(1e-6)
    yt = (y_train - mean) / std
    yv = (y_val - mean) / std

    model = FlickEncoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    gen = torch.Generator().manual_seed(seed)
    mse = torch.nn.functional.mse_loss

    xv_dev, yv_dev = ds.x_val.to(dev), yv.to(dev)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state: dict | None = None
    best_heads: dict[str, float] = {}
    best_epoch = 0
    bad = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        total = 0.0
        seen = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            if idx.numel() < 2:  # BatchNorm needs >1 sample in train mode
                continue
            xb, yb = ds.x_train[idx].to(dev), yt[idx].to(dev)
            pred, _ = model(xb)
            loss = mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.numel()
            seen += idx.numel()
        sched.step()
        train_loss = total / max(seen, 1)

        model.eval()
        with torch.no_grad():
            pred_v, _ = model(xv_dev)
            val_loss = float(mse(pred_v, yv_dev))
            per_head = {
                name: float(mse(pred_v[:, i], yv_dev[:, i]))
                for i, name in enumerate(TRAIN_HEADS)
            }
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_heads = per_head
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / CHECKPOINT_NAME
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": model.config,
            "target_mean": mean.cpu(),
            "target_std": std.cpu(),
            "heads": list(TRAIN_HEADS),
            "n_samples": n_samples,
            "rate": rate,
        },
        ckpt_path,
    )
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "device": dev,
        "seed": seed,
        "params": count_parameters(model),
        "n_traces": ds.n_traces,
        "n_flicks": ds.n_flicks,
        "train_size": n_train,
        "val_size": n_val,
        "epochs_requested": epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "val_loss_per_head": best_heads,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "history": history,
    }
    meta_path = out / METADATA_NAME
    meta_path.write_text(json.dumps(meta, indent=2))

    return TrainResult(
        checkpoint=ckpt_path,
        metadata=meta_path,
        device=dev,
        params=meta["params"],
        n_traces=ds.n_traces,
        train_size=n_train,
        val_size=n_val,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        train_loss=meta["final_train_loss"],
        val_loss=best_val,
        val_loss_per_head=best_heads,
        history=history,
    )
