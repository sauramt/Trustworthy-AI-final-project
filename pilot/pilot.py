"""Pilot for the TAI Group 3 project: uncertainty-adaptive confidence loss.

Pipeline (one seed):
  1. Fixed stratified split of the CIFAR-10 training set: teacher / student / validation.
  2. Weak teacher: small CNN trained on the teacher set with true labels.
  3. Strong student: frozen DINO ViT-S/16 features (cached) + small MLP head.
  4. Training strategies on the teacher's labels:
       A        naive (alpha = 0)
       B        A02 confidence loss, fixed alpha
       C        ours, alpha_i = alpha_max * (1 - u_i), u_i from MC Dropout
       Cnorm    ours, rescaled: alpha_i = alpha_max * (1 - u_i) / mean(1 - u), capped at 1
       Bprime   fixed alpha equal to the mean alpha_i used by C
       D25/50/75  fixed alpha, gated self-target: the fraction q of each batch with the
                lowest model uncertainty (mutual information, MC Dropout) gets a hard
                target argmax(mean p), the rest keeps the soft mean p
       Drand50  same as D50 with a random gate (control)
       ceiling  same head trained on true labels
  5. Evaluation on the CIFAR-10 test set: accuracy, PGR, ECE, accuracy at 80/90 %
     coverage, AUROC for error detection (all errors, and errors copied from the teacher).

On the NYCU cluster (compute nodes may have no Internet):
  login node : python pilot.py --prepare     (downloads CIFAR-10 and DINO, no training)
  then       : sbatch run_pilot.sbatch
Check anywhere: python pilot.py --smoke      (synthetic data, CPU, 10 seconds, no download)
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10


# ----------------------------------------------------------------------------- data
def stratified_split(labels, sizes, seed):
    """Split indices into consecutive parts of the given sizes, stratified by class."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in sizes]
    total = sum(sizes)
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.where(labels == c)[0])
        start = 0
        for k, size in enumerate(sizes):
            n = round(len(idx) * size / total)
            parts[k].append(idx[start:start + n])
            start += n
    return [np.sort(np.concatenate(p)) for p in parts]


def load_cifar(root):
    from torchvision.datasets import CIFAR10
    train = CIFAR10(root, train=True, download=True)
    test = CIFAR10(root, train=False, download=True)
    return (train.data, np.array(train.targets)), (test.data, np.array(test.targets))


# -------------------------------------------------------------------------- teacher
class SmallCNN(nn.Module):
    """AlexNet-style network for 32x32 images."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, NUM_CLASSES))

    def forward(self, x):
        return self.classifier(self.features(x))


def to_tensor_32(images, device):
    """uint8 NHWC array -> normalised float NCHW tensor on the device."""
    x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def train_teacher(images, labels, epochs, device, seed):
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = to_tensor_32(images, device), torch.from_numpy(labels).to(device)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(x), device=device).split(128):
            opt.zero_grad()
            F.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    return model


@torch.no_grad()
def teacher_predict(model, images, device):
    model.eval()
    x = to_tensor_32(images, device)
    return torch.cat([model(x[i:i + 1000]).argmax(1) for i in range(0, len(x), 1000)]).cpu().numpy()


# ------------------------------------------------------------------------- features
def load_dino():
    """DINO ViT-S/16 from torch.hub. Uses the local cache when there is one (no network)."""
    return torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                          trust_repo=True, skip_validation=True)


@torch.no_grad()
def dino_features(images, device, batch=128):
    """CLS features (384-d) of the frozen DINO ViT-S/16, self-supervised on ImageNet-1k."""
    model = load_dino().to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            f = model((x - mean) / std)
        out.append(f.float().cpu())
        if (i // batch) % 50 == 0:
            print(f"  features {i}/{len(images)}", flush=True)
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- head
class Head(nn.Module):
    def __init__(self, dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, NUM_CLASSES))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def mc_dropout_probs(model, x, passes):
    """Mean softmax over several stochastic passes (dropout active)."""
    was_training = model.training
    model.train()
    probs = torch.stack([F.softmax(model(x), 1) for _ in range(passes)]).mean(0)
    model.train(was_training)
    return probs


@torch.no_grad()
def mc_dropout_stats(model, x, passes):
    """Mean softmax and model uncertainty over several stochastic passes.

    Model (epistemic) uncertainty = entropy of the mean - mean entropy of the passes
    (the mutual information used in U09), in nats.
    """
    was_training = model.training
    model.train()
    probs = torch.stack([F.softmax(model(x), 1) for _ in range(passes)])
    model.train(was_training)
    mean = probs.mean(0)
    entropy = lambda p: -(p * p.clamp_min(1e-12).log()).sum(-1)
    return mean, (entropy(mean) - entropy(probs).mean(0)).clamp_min(0)


def normalised_entropy(probs):
    """Predictive entropy divided by log(number of classes): 0 = certain, 1 = uniform."""
    return -(probs * probs.clamp_min(1e-12).log()).sum(1) / math.log(NUM_CLASSES)


def train_head(mode, x, y, x_val, y_val, args, device, fixed_alpha=None, q=None, y_true=None):
    """Train one student head.

    mode: "A" (alpha = 0), "B" (fixed alpha), "C" (alpha per image), "Cnorm" (alpha per image,
    rescaled so that its batch mean equals alpha_max), "ceiling" (true labels, alpha = 0). y and y_val are the labels the student is allowed to see.
    "D": fixed alpha as in B, but the student's own prediction is hardened only on the share q
    of each batch with the lowest model uncertainty; elsewhere the target is the soft MC mean.
    "Drand": same, with a random gate of the same share q (control).
    y_true (never used for training) only feeds the gate diagnostic of D and Drand.
    Returns the best model by validation accuracy and a dict with the mean alpha used after
    warm-up and, for D and Drand, the teacher-error rate on hardened and on soft images.
    """
    torch.manual_seed(args.seed)
    model = Head(x.shape[1], dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_total = args.epochs * math.ceil(len(x) / args.batch)
    warmup = max(1, int(args.warmup * steps_total))
    alpha_max = fixed_alpha if fixed_alpha is not None else args.alpha_max
    step, best_acc, best_state, alphas = 0, -1.0, None, []
    gate = {"hard_n": 0, "hard_err": 0, "soft_n": 0, "soft_err": 0}
    for _ in range(args.epochs):
        model.train()
        for b in torch.randperm(len(x), device=device).split(args.batch):
            ramp = min(1.0, step / warmup)
            logits = model(x[b])
            if mode in ("A", "ceiling"):
                alpha = torch.zeros(len(b), device=device)
            elif mode in ("B", "D", "Drand"):
                alpha = torch.full((len(b),), alpha_max * ramp, device=device)
            elif mode in ("C", "Cnorm"):
                u = normalised_entropy(mc_dropout_probs(model, x[b], args.mc_train))
                w = 1.0 - u
                if mode == "Cnorm":                  # same mean alpha as B in every batch
                    w = w / w.mean().clamp_min(1e-6)
                alpha = (alpha_max * ramp * w).clamp(max=1.0)
            else:
                raise ValueError(mode)
            if mode in ("D", "Drand"):
                mean, u = mc_dropout_stats(model, x[b], args.mc_train)
                n_hard = int(round(q * len(b)))
                order = u.argsort() if mode == "D" else torch.randperm(len(b), device=device)
                hard = torch.zeros(len(b), dtype=torch.bool, device=device)
                hard[order[:n_hard]] = True          # lowest model uncertainty (or random)
                target = torch.where(hard.unsqueeze(1),
                                     F.one_hot(mean.argmax(1), NUM_CLASSES).float(), mean)
                self_loss = -(target * F.log_softmax(logits, 1)).sum(1)
                if y_true is not None and ramp >= 1.0:
                    err = y[b] != y_true[b]          # teacher mislabelled this image
                    gate["hard_n"] += int(hard.sum()); gate["hard_err"] += int((err & hard).sum())
                    gate["soft_n"] += int((~hard).sum()); gate["soft_err"] += int((err & ~hard).sum())
            else:
                own = logits.detach().argmax(1)      # the student's own hardened prediction
                self_loss = F.cross_entropy(logits, own, reduction="none")
            loss = ((1 - alpha) * F.cross_entropy(logits, y[b], reduction="none")
                    + alpha * self_loss).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ramp >= 1.0:
                alphas.append(alpha.mean().item())
            step += 1
        model.eval()
        with torch.no_grad():
            acc = (model(x_val).argmax(1) == y_val).float().mean().item()
        if acc > best_acc:                           # early stopping on the labels we may see
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    info = {"mean_alpha": float(np.mean(alphas)) if alphas else 0.0}
    if gate["hard_n"] and gate["soft_n"]:
        info["teacher_err_hardened"] = gate["hard_err"] / gate["hard_n"]
        info["teacher_err_soft"] = gate["soft_err"] / gate["soft_n"]
    return model, info


# -------------------------------------------------------------------------- metrics
def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def accuracy_at_coverage(uncertainty, correct, coverage):
    """Accuracy on the `coverage` share of images with the lowest uncertainty."""
    keep = np.argsort(uncertainty, kind="stable")[:int(round(coverage * len(correct)))]
    return float(correct[keep].mean())


def auroc(score, positive):
    """Probability that a positive has a higher score than a negative (ties count half)."""
    positive = positive.astype(bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    sorted_score = score[order]
    start = 0                                        # average the ranks of tied scores
    for end in range(1, len(score) + 1):
        if end == len(score) or sorted_score[end] != sorted_score[start]:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, x_test, y_test, teacher_test, args):
    model.eval()
    soft = F.softmax(model(x_test), 1)
    mc, u_model = mc_dropout_stats(model, x_test, args.mc_test)
    pred = soft.argmax(1).cpu().numpy()
    correct = (pred == y_test).astype(float)
    u_soft = (1 - soft.max(1).values).cpu().numpy()  # uncertainty from max softmax
    u_mc = normalised_entropy(mc).cpu().numpy()      # uncertainty from MC Dropout entropy
    wrong = correct == 0
    copied = wrong & (pred == teacher_test)          # errors identical to the teacher's
    keep = (correct == 1) | copied
    res = {
        "accuracy": float(correct.mean()),
        "ece": ece(soft.max(1).values.cpu().numpy(), correct),
        "share_of_errors_copied": float(copied.sum() / max(1, wrong.sum())),
    }
    for name, u in (("softmax", u_soft), ("mc", u_mc), ("model", u_model.cpu().numpy())):
        res[f"acc@80_{name}"] = accuracy_at_coverage(u, correct, 0.8)
        res[f"acc@90_{name}"] = accuracy_at_coverage(u, correct, 0.9)
        res[f"auroc_errors_{name}"] = auroc(u, wrong)
        res[f"auroc_copied_{name}"] = auroc(u[keep], copied[keep])
    return res


# ----------------------------------------------------------------------------- main
def synthetic_data(seed):
    """Small stand-in for the real pipeline, used by --smoke to test the logic on CPU."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1.0, (NUM_CLASSES, 64))

    def make(n):
        y = rng.integers(0, NUM_CLASSES, n)
        x = centres[y] + rng.normal(0, 2.2, (n, 64))
        confuse = rng.random(n) < np.where(y % 2 == 0, 0.45, 0.10)   # class-dependent errors
        weak = np.where(confuse, (y + 1) % NUM_CLASSES, y)
        return x.astype(np.float32), y, weak

    return {"student": make(4000), "val": make(1000), "test": make(3000)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher-epochs", type=int, default=3,
                   help="fewer epochs = weaker teacher; aim for 60 to 75 %% test accuracy")
    p.add_argument("--alpha-max", type=float, default=0.5)
    p.add_argument("--warmup", type=float, default=0.2, help="share of steps to ramp alpha up")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--mc-train", type=int, default=10)
    p.add_argument("--mc-test", type=int, default=20)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--prepare", action="store_true",
                   help="only download CIFAR-10 and the DINO weights, then exit")
    args = p.parse_args()

    if args.prepare:
        load_cifar(args.data)
        load_dino()
        print("CIFAR-10 and DINO ViT-S/16 are downloaded. Now run: sbatch run_pilot.sbatch")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    tag = "smoke" if args.smoke else f"seed{args.seed}_T{args.teacher_epochs}"
    print(f"device: {device}")

    if args.smoke:
        d = synthetic_data(args.seed)
        args.epochs = 8
        teacher_acc = float((d["test"][2] == d["test"][1]).mean())
    else:
        cache = os.path.join(args.out, f"cache_{tag}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            d = {k: (z[f"{k}_x"], z[f"{k}_y"], z[f"{k}_weak"]) for k in ("student", "val", "test")}
            teacher_acc = float(z["teacher_acc"])
            print(f"loaded cache {cache}")
        else:
            (tr_img, tr_y), (te_img, te_y) = load_cifar(args.data)
            i_teacher, i_student, i_val = stratified_split(tr_y, [20000, 25000, 5000], args.seed)
            np.savez(os.path.join(args.out, f"split_seed{args.seed}.npz"),
                     teacher=i_teacher, student=i_student, val=i_val)
            print("training the teacher")
            teacher = train_teacher(tr_img[i_teacher], tr_y[i_teacher], args.teacher_epochs,
                                    device, args.seed)
            weak = {k: teacher_predict(teacher, img, device) for k, img in
                    (("student", tr_img[i_student]), ("val", tr_img[i_val]), ("test", te_img))}
            teacher_acc = float((weak["test"] == te_y).mean())
            print("extracting DINO ViT-S/16 features")
            d = {"student": (dino_features(tr_img[i_student], device), tr_y[i_student], weak["student"]),
                 "val": (dino_features(tr_img[i_val], device), tr_y[i_val], weak["val"]),
                 "test": (dino_features(te_img, device), te_y, weak["test"])}
            np.savez(cache, teacher_acc=teacher_acc,
                     **{f"{k}_{n}": v for k, t in d.items() for n, v in zip(("x", "y", "weak"), t)})

    mu, sd = d["student"][0].mean(0), d["student"][0].std(0) + 1e-6
    t = lambda a, dtype=None: torch.as_tensor(a, dtype=dtype, device=device)
    xs, xv, xt = (t((d[k][0] - mu) / sd, torch.float32) for k in ("student", "val", "test"))
    ys_true, ys_weak = t(d["student"][1], torch.long), t(d["student"][2], torch.long)
    yv_true, yv_weak = t(d["val"][1], torch.long), t(d["val"][2], torch.long)
    y_test, teacher_test = np.asarray(d["test"][1]), np.asarray(d["test"][2])
    print(f"teacher test accuracy: {teacher_acc:.4f}   "
          f"weak-label accuracy on the student set: {(ys_weak == ys_true).float().mean().item():.4f}")

    results, alpha_c = {}, None
    for name in ("A", "B", "C", "Cnorm", "Bprime", "D25", "D50", "D75", "Drand50", "ceiling"):
        if name == "ceiling":
            model, info = train_head("ceiling", xs, ys_true, xv, yv_true, args, device)
        elif name == "Bprime":
            model, info = train_head("B", xs, ys_weak, xv, yv_weak, args, device,
                                     fixed_alpha=alpha_c)
        elif name.startswith("D"):
            mode = "Drand" if name.startswith("Drand") else "D"
            model, info = train_head(mode, xs, ys_weak, xv, yv_weak, args, device,
                                     q=int(name[len(mode):]) / 100, y_true=ys_true)
        else:
            model, info = train_head(name, xs, ys_weak, xv, yv_weak, args, device)
            if name == "C":
                alpha_c = info["mean_alpha"]
        results[name] = evaluate(model, xt, y_test, teacher_test, args)
        results[name].update(info)

    gap = results["ceiling"]["accuracy"] - teacher_acc
    for name in results:
        results[name]["pgr"] = (results[name]["accuracy"] - teacher_acc) / gap if gap > 0 else float("nan")

    cols = ["accuracy", "pgr", "ece", "acc@80_softmax", "acc@80_mc", "acc@90_mc",
            "auroc_errors_mc", "auroc_errors_model", "auroc_copied_mc", "mean_alpha"]
    print("\n" + "model".ljust(9) + "".join(c.rjust(16) for c in cols))
    for name, r in results.items():
        print(name.ljust(9) + "".join(f"{r[c]:16.4f}" for c in cols))
    print("\ngate check: share of wrong teacher labels among hardened / soft examples")
    for name, r in results.items():
        if "teacher_err_hardened" in r:
            print(f"{name.ljust(9)}{r['teacher_err_hardened']:16.4f}{r['teacher_err_soft']:16.4f}")
    path = os.path.join(args.out, f"results_{tag}.json")
    with open(path, "w") as f:
        json.dump({"teacher_accuracy": teacher_acc, "args": vars(args), "results": results}, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
