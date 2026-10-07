"""Train one model:  python train.py --model svr|nn|seq2seq"""
import argparse
import time

import numpy as np
import torch

import config as C
import models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=models.NAMES)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kernel", default="poly", choices=["poly", "linear"])
    ap.add_argument("--n_val", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(C.DATA_DIR / "dataset.npz")
    X, Y = d["X"], d["Y"]
    nv = args.n_val
    Xtr, Ytr, Xva, Yva = X[:-nv], Y[:-nv], X[-nv:], Y[-nv:]
    print(f"train {len(Xtr)}  val {len(Xva)}  device {device}")

    model = models.build(args.model, kernel=args.kernel)
    losses = []
    t0 = time.time()

    if args.model == "svr":
        model.fit(Xtr.astype(np.float64), Ytr)
    else:
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        Xt = torch.as_tensor(Xtr, dtype=torch.long, device=device)
        Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(Xt), device=device)
            ep_loss = []
            for i in range(0, len(Xt), args.batch):
                idx = perm[i:i + args.batch]
                pred = model(Xt[idx], Yt[idx])
                loss = torch.mean((pred - Yt[idx]) ** 2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
                ep_loss.append(loss.item())
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"epoch {ep + 1:3d}/{args.epochs}  loss {np.mean(ep_loss):.5f}  "
                      f"({time.time() - t0:.0f}s)")

    p = models.save(args.model, model)
    print("saved ->", p)

    # validation: MSE on held-out sentences vs. a trivial "mean spectrogram" baseline
    pred = models.predict(args.model, model, Xva, device)
    mse = float(np.mean((pred - Yva) ** 2))
    base = float(np.mean((Ytr.mean(0, keepdims=True) - Yva) ** 2))
    print(f"val MSE {mse:.5f}   (mean-spectrogram baseline {base:.5f})")
    with open(C.RESULTS_DIR / "val_mse.txt", "a") as f:
        f.write(f"{args.model}\t{mse:.6f}\tbaseline {base:.6f}\n")

    if losses:
        np.save(C.RESULTS_DIR / f"loss_{args.model}.npy", np.array(losses))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(losses)
        plt.xlabel("iteration")
        plt.ylabel("training MSE loss")
        plt.title(f"{args.model} training loss")
        plt.tight_layout()
        plt.savefig(C.RESULTS_DIR / f"loss_{args.model}.png", dpi=150)
        print("loss curve ->", C.RESULTS_DIR / f"loss_{args.model}.png")


if __name__ == "__main__":
    main()
