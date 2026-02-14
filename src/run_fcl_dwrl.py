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
import csv
import json
import pandas as pd


from src.fl import Server, Client
from src.strategies.replay import ReplayBuffer

# use your single source of truth
from src.data.dwrl_labels import NUM_CLASSES, IDX2CLASS
from src.data.dwrl_dataset import DWRLDataset   # you created this earlier
from src.data.dwrl_transforms import make_transforms  # must exist and return (train_tf, eval_tf)
from src.policy_dwrl import V4ControllerDWRL


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

@torch.no_grad()
def eval_acc_and_recall(model, loader, device, num_classes: int):
    model.eval()
    correct, total = 0, 0
    hits = torch.zeros(num_classes, dtype=torch.long)
    counts = torch.zeros(num_classes, dtype=torch.long)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)

        correct += (pred == yb).sum().item()
        total += yb.numel()

        for c in range(num_classes):
            m = (yb == c)
            if m.any():
                counts[c] += m.sum().item()
                hits[c] += (pred[m] == c).sum().item()

    acc = 100.0 * correct / max(1, total)
    recall = (hits.float() / counts.clamp(min=1).float()).cpu().numpy()  # [0..1]
    return acc, recall

@torch.no_grad()
def model_divergence_norm(global_model, client_models, device):
    # simple scalar: std(dist) / median(dist)
    # dist = L2 norm between flattened params
    gvec = torch.cat([p.detach().to(device).flatten() for p in global_model.parameters()])
    dists = []
    for m in client_models:
        cvec = torch.cat([p.detach().to(device).flatten() for p in m.parameters()])
        d = torch.norm(cvec - gvec, p=2).item()
        dists.append(d)
    if len(dists) < 2:
        return 0.0
    med = float(np.median(dists)) + 1e-8
    return float(np.std(dists) / med)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(1).cpu().tolist()
        ps.extend(pred)
        ys.extend(yb.tolist())
    return ys, ps

def confusion_matrix_np(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def save_artifacts(run_dir: Path, args, global_model, val_acc_pct: float, test_acc_pct: float, test_loader, device):
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) save global model
    ckpt = {
        "state_dict": global_model.state_dict(),
        "args": vars(args),
        "val_acc_pct": float(val_acc_pct),
        "test_acc_pct": float(test_acc_pct),
    }
    torch.save(ckpt, run_dir / "global_model.pt")

    # 2) save args
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # 3) per-class acc + confusion on TEST
    y_true, y_pred = predict(global_model, test_loader, device)

    # per-class accuracy (not recall) on TEST
    percls = {}
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    for i in range(NUM_CLASSES):
        m = (y_true_np == i)
        percls[IDX2CLASS[i]] = float((y_pred_np[m] == i).mean()) if m.any() else float("nan")
    (run_dir / "per_class_acc.json").write_text(json.dumps(percls, indent=2))

    cm = confusion_matrix_np(y_true, y_pred, NUM_CLASSES)
    np.save(run_dir / "confusion.npy", cm)

    # also CSV
    df_cm = pd.DataFrame(cm,
                         index=[IDX2CLASS[i] for i in range(NUM_CLASSES)],
                         columns=[IDX2CLASS[i] for i in range(NUM_CLASSES)])
    df_cm.to_csv(run_dir / "confusion.csv")


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
    ap.add_argument("--controller", choices=["fixed", "v4"], default="fixed")
    ap.add_argument("--log_csv", type=str, default="")  # if empty -> run_dir/round_logs.csv
    ap.add_argument("--run_dir", type=str, default="")  # where to save ckpt/metrics

    # CL / client split config
    ap.add_argument("--split_mode", choices=["equal", "dirichlet"], default="equal")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--cl_batches", type=int, default=5)
    
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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


    # logging
    run_dir = Path(args.run_dir) if args.run_dir else (Path("runs") / f"fcl_seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv) if args.log_csv else (run_dir / "round_logs.csv")

    round_rows = []
    best_recall = np.zeros(NUM_CLASSES, dtype=np.float32)

    # controller state (fractions, not %)
    # ctrl_acc_curr: most recent test acc fraction; ctrl_acc_prev: one step before that.
    ctrl_acc_curr = None
    ctrl_acc_prev = None
    last_forget_mean = 0.0
    last_divergence = 0.0

    v4 = None

    if args.controller == "v4":
        v4 = V4ControllerDWRL(base_lr=args.lr, base_rep=args.replay_ratio)
    for r in range(args.rounds):

        # -------- controller decision (GLOBAL) --------
        if args.controller == "v4":
            hp = v4.step(
                round_id=r,
                acc=float(ctrl_acc_curr) if ctrl_acc_curr is not None else 0.0,
                last_acc=ctrl_acc_prev,
                global_val_loss=None,
                forget_mean=last_forget_mean,
                divergence=last_divergence,
            )
        else:
            hp = {
                "lr": args.lr,
                "replay_ratio": args.replay_ratio,
                "lr_req": args.lr,
                "replay_req": args.replay_ratio,
                "clamped_lr": False,
                "clamped_rep": False,
                "notes": "fixed",
            }

        print(
            f"[HP r={r}] controller={args.controller} "
            f"lr={hp['lr']:.6f} replay={hp['replay_ratio']:.2f} "
            f"(req_lr={hp.get('lr_req', hp['lr']):.6f}, "
            f"req_rep={hp.get('replay_req', hp['replay_ratio']):.2f}, "
            f"clamped_lr={bool(hp.get('clamped_lr', False))}, "
            f"clamped_rep={bool(hp.get('clamped_rep', False))}) "
            f"({hp['notes']})",
            flush=True,
        )
        # broadcast
        for c in clients:
            c.load_state_from(global_model)
            for pg in c.optimizer.param_groups:
                pg["lr"] = float(hp["lr"])


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
                    replay_ratio=hp["replay_ratio"],
                    epoch=e,
                    total_epochs=args.epochs,
                    log_interval=200,
                )
                if stop:
                    break

        # aggregate
        global_model = server.average([c.model for c in clients])

        # val acc (percent)
        acc_val = eval_acc(global_model, val_loader, device)

        # test acc + per-class recall
        acc_test_pct, per_class_recall = eval_acc_and_recall(global_model, test_loader, device, NUM_CLASSES)

        # forgetting (mean)
        forgetting = np.maximum(0.0, best_recall - per_class_recall)
        best_recall = np.maximum(best_recall, per_class_recall)
        forget_mean = float(np.mean(forgetting))

        # divergence (scalar)
        div_norm = model_divergence_norm(global_model, [c.model for c in clients], device)

        print(f"[Round {r}] val_acc={acc_val:.2f}% test_acc={acc_test_pct:.2f}% forget_mean={forget_mean:.4f} div={div_norm:.4f}", flush=True)

        # update history for controller (fractions)
        ctrl_acc_prev = ctrl_acc_curr
        ctrl_acc_curr = float(acc_test_pct / 100.0)
        last_forget_mean = float(forget_mean)
        last_divergence = float(div_norm)

        # csv row
        round_rows.append({
            "round": int(r),
            "controller": str(args.controller),
            "lr": float(hp["lr"]),
            "replay_ratio": float(hp["replay_ratio"]),
            "lr_req": float(hp.get("lr_req", hp["lr"])),
            "replay_req": float(hp.get("replay_req", hp["replay_ratio"])),
            "clamped_lr": bool(hp.get("clamped_lr", False)),
            "clamped_rep": bool(hp.get("clamped_rep", False)),
            "val_acc_pct": float(acc_val),
            "test_acc_pct": float(acc_test_pct),
            "forget_mean": float(forget_mean),
            "divergence": float(div_norm),
            "notes": str(hp.get("notes", "")),
        })

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


    # write round logs
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(round_rows[0].keys()) if round_rows else ["round"])
        w.writeheader()
        for row in round_rows:
            w.writerow(row)
    print(f"[Saved] round_logs.csv -> {log_csv}", flush=True)

    # final artifacts (model + per-class + confusion)
    # use last computed accs if available, otherwise evaluate once
    final_val = eval_acc(global_model, val_loader, device)
    final_test, _ = eval_acc_and_recall(global_model, test_loader, device, NUM_CLASSES)
    save_artifacts(run_dir, args, global_model, final_val, final_test, test_loader, device)
    print(f"[Saved] {run_dir}/global_model.pt + per_class_acc.json + confusion.csv", flush=True)


if __name__ == "__main__":
    main()
