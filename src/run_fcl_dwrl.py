# src/run_fcl_dwrl.py
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import models
from pathlib import Path

from src.fl import Server, Client
from src.strategies.replay import ReplayBuffer

# use your single source of truth
from src.data.dwrl_labels import NUM_CLASSES
from src.data.dwrl_dataset import DWRLDataset   # you created this earlier
from src.data.dwrl_transforms import make_transforms  # must exist and return (train_tf, eval_tf)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "src" / "data" / "splits"


def build_resnet18(num_classes: int):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / max(1, total)


def pick_device(name: str):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=1)          # ✅ start with 1
    ap.add_argument("--epochs", type=int, default=1)          # ✅ start with 1
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replay_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--train_csv", type=str, default=str(SPLITS_DIR / "train_plus_aug.csv"))
    ap.add_argument("--val_csv", type=str, default=str(SPLITS_DIR / "val_plus_aug.csv"))
    ap.add_argument("--test_csv", type=str, default=str(SPLITS_DIR / "test.csv"))
    ap.add_argument("--no_val", action="store_true")
    ap.add_argument("--early_round_patience", type=int, default=3)
    ap.add_argument("--early_round_delta", type=float, default=0.2)  # in % points (0.2 = 0.2%)

        # CL / client split config
    ap.add_argument("--split_mode", choices=["equal", "dirichlet"], default="equal")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--cl_batches", type=int, default=5)
    
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    best_val = -1e9
    bad_rounds = 0      

    device = pick_device(args.device)
    print(f"[Device] {device}", flush=True)

    train_tf, eval_tf = make_transforms(224)

    # datasets (drop_missing=True avoids crashing if 1 file is missing)
    train_ds = DWRLDataset(args.train_csv, transform=train_tf, verify_paths=True, drop_missing=True)
    val_ds   = DWRLDataset(args.val_csv,   transform=eval_tf,  verify_paths=True, drop_missing=True)
    test_ds  = DWRLDataset(args.test_csv,  transform=eval_tf,  verify_paths=True, drop_missing=True)


    print(f"[Data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)


    # quick sanity batch
    xb, yb = next(iter(DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)))
    print(f"[Sanity] batch x={tuple(xb.shape)} y={yb.tolist()}", flush=True)

    # ---------------------------
    # Client split + CL schedule
    # ---------------------------
    from src.data.dwrl_labels import CLASS2IDX
    from src.data.split_clients_dwrl import make_client_splits, make_cl_schedule

    targets = np.array(
        [CLASS2IDX[s] for s in train_ds.df["label"].tolist()],
        dtype=np.int64
    )

    splits = make_client_splits(
        n=len(train_ds),
        targets=targets,              # used only if split_mode="dirichlet"
        n_clients=args.clients,
        mode=args.split_mode,         # "equal" or "dirichlet"
        alpha=args.alpha,
        seed=args.seed,
    )

    cl_schedule = make_cl_schedule(
        splits,
        cl_batches=args.cl_batches,
        seed=args.seed,
    )

    for cid in range(args.clients):
        sizes = [len(b) for b in cl_schedule[cid]]
        print(f"[CL] client {cid}: {sizes} (sum={sum(sizes)})", flush=True)




    # shared val/test loaders
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)

    clients = []
    for cid in range(args.clients):
        # start with CL batch 0
        init_idx = cl_schedule[cid][0]
        subset = Subset(train_ds, init_idx.tolist())
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        model = build_resnet18(NUM_CLASSES).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr)
        replay = ReplayBuffer(capacity=2000)

        clients.append(Client(
            cid=cid,
            model=model,
            optimizer=opt,
            train_loader=loader,
            device=device,
            replay=replay,
            val_loader=val_loader,
            early_patience=2,
        ))
        print(f"[InitSplit] client {cid}: batch1={len(subset)} (of total {sum(len(b) for b in cl_schedule[cid])})", flush=True)
        server = Server(device=device)

    # init global
    global_model = server.average([c.model for c in clients])
    base = eval_acc(global_model, test_loader, device)
    print(f"[Init] test_acc={base:.2f}%", flush=True)

    for r in range(args.rounds):
        # broadcast
        for c in clients:
            c.load_state_from(global_model)


         # ✅ set current CL batch loader for this round (BEFORE training)
        b_id = min(r, args.cl_batches - 1)  # safe if rounds > cl_batches
        for c in clients:
            batch_idx = cl_schedule[c.cid][b_id]
            c.loader = DataLoader(
                Subset(train_ds, batch_idx.tolist()),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            print(
                f"[Round {r}] client {c.cid}: CL_batch={b_id+1}/{args.cl_batches} new={len(batch_idx)}",
                flush=True,
            )

        # local train
        for c in clients:
            for e in range(args.epochs):
                _, _, stop = c.train_one_epoch(
                    replay_ratio=args.replay_ratio,
                    epoch=e,
                    total_epochs=args.epochs,
                    log_interval=200,
                )
                if stop:
                    break

        # aggregate
        global_model = server.average([c.model for c in clients])
        acc_val = eval_acc(global_model, val_loader, device)
        acc_test = eval_acc(global_model, test_loader, device)
        print(f"[Round {r}] val_acc={acc_val:.2f}% test_acc={acc_test:.2f}%", flush=True)

        # --- round-level early stopping on GLOBAL val_acc ---
        if acc_val > best_val + args.early_round_delta:
            best_val = acc_val
            bad_rounds = 0
        else:
            bad_rounds += 1
            print(f"[EarlyStop] no improv for {bad_rounds}/{args.early_round_patience} rounds (best_val={best_val:.2f}%)", flush=True)
            if bad_rounds >= args.early_round_patience:
                print(f"[EarlyStop] stopping at round {r} (best_val={best_val:.2f}%)", flush=True)
                break


if __name__ == "__main__":
    main()