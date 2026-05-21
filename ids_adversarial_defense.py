import argparse
import time
import warnings
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.neighbors import KernelDensity

warnings.filterwarnings("ignore")

# =============================================================================
# 0. REPRODUCIBILITY & GLOBAL CONFIG
# =============================================================================

SEED = 42

# -----------------------------------------------------------------------------
#  YOUR DATASET PATH  (edit this line if your file moves)
# -----------------------------------------------------------------------------
DATASET_PATH = r"C:\Users\HP\Downloads\btp\data\cic1.csv"
# -----------------------------------------------------------------------------


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def load_cicids2018(csv_path=DATASET_PATH):
    """
    Load the CSE-CIC-IDS2018 dataset from csv_path.

    If the file is not found, a synthetic 78-feature proxy is generated
    automatically so you can test the pipeline without the real data.

    Expected CSV format
    -------------------
    - Column 'Label'   : 'Benign' or attack name  ->  encoded as 0 / 1
    - Drop columns     : 'Fwd Pkt Len Std', 'Timestamp'  (per paper Sec 5)
    - Remaining columns: 78 numeric features

    Returns
    -------
    X_train, X_test : np.ndarray  float32, StandardScaler-normalised
    y_train, y_test : np.ndarray  int64, binary (0 = benign, 1 = attack)
    n_features      : int
    """

    if csv_path and os.path.exists(csv_path):
        import pandas as pd

        print(f"[Data] Loading dataset from:\n       {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)

        # ---- detect label column (case-insensitive) -------------------------
        label_col = None
        for col in df.columns:
            if col.strip().lower() == "label":
                label_col = col
                break

        if label_col is None:
            print("[Data] WARNING: No 'Label' column found.")
            print(f"[Data] Columns in your file: {df.columns.tolist()[:15]}")
            print("[Data] Falling back to synthetic proxy data.")
            csv_path = None
        else:
            # encode: benign -> 0, everything else -> 1
            df[label_col] = (
                df[label_col].astype(str).str.strip().str.lower() != "benign"
            ).astype(int)

            # drop columns specified in paper Section 5
            drop_cols = [
                c for c in df.columns
                if c.strip() in ("Fwd Pkt Len Std", "Timestamp")
            ]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                print(f"[Data] Dropped columns: {drop_cols}")

            # clean infinities and NaNs
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            y = df[label_col].values.astype(np.int64)
            X = (
                df.drop(columns=[label_col])
                  .select_dtypes(include=[np.number])
                  .values
                  .astype(np.float32)
            )

            print(f"[Data] Loaded {len(X):,} rows  |  {X.shape[1]} features")
            print(f"[Data] Attack rate: {y.mean():.2%}")

    # fallback: synthetic proxy data
    if not csv_path or not os.path.exists(csv_path):
        print("[Data] Generating synthetic 78-feature proxy dataset ...")
        X, y = make_classification(
            n_samples=12_000,
            n_features=78,
            n_informative=40,
            n_redundant=15,
            n_clusters_per_class=3,
            weights=[0.64, 0.36],   # mirrors paper class distribution
            flip_y=0.01,
            random_state=SEED,
        )
        X = X.astype(np.float32)
        y = y.astype(np.int64)

    # train / test split  (80 / 20  as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # StandardScaler normalisation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    n_features = X_train.shape[1]
    print(
        f"[Data] Train: {X_train.shape}  |  Test: {X_test.shape}  |  "
        f"Features: {n_features}  |  Attack rate (train): {y_train.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test, n_features


# =============================================================================
# 2. DEEP NEURAL NETWORK  (matches paper Section 5.1 architecture)
# =============================================================================

class IDSClassifier(nn.Module):
    """
    Input -> Linear(256) -> ReLU -> Dropout(0.5)
          -> Linear(128) -> ReLU -> Dropout(0.5)
          -> Linear(1)   -> Sigmoid
    """

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        return (self.forward(x) >= threshold).long()


# =============================================================================
# 3. TRAINING & EVALUATION
# =============================================================================

def train_model(model, X_train, y_train,
                epochs=30, batch_size=512, lr=1e-3, verbose=True):
    """Adam + BCELoss training loop. Returns per-epoch accuracy list."""

    dataset = TensorDataset(
        torch.from_numpy(X_train).to(DEVICE),
        torch.from_numpy(y_train).float().to(DEVICE),
    )
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    model.train()
    history = []

    for epoch in range(1, epochs + 1):
        correct = total = 0
        for xb, yb in loader:
            optim.zero_grad()
            preds = model(xb)
            loss  = loss_fn(preds, yb)
            loss.backward()
            optim.step()
            correct += ((preds >= 0.5).long() == yb.long()).sum().item()
            total   += yb.size(0)

        acc = correct / total
        history.append(acc)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>3}/{epochs}  train_acc = {acc:.4f}")

    return history


def evaluate_model(model, X, y, label="Evaluation"):
    """Compute accuracy, precision, recall, F1 and print a summary."""

    model.eval()
    with torch.no_grad():
        preds = model.predict(torch.from_numpy(X).to(DEVICE)).cpu()
    yt = torch.from_numpy(y)

    acc  = accuracy_score(yt, preds)
    prec = precision_score(yt, preds, zero_division=0)
    rec  = recall_score(yt, preds, zero_division=0)
    f1   = f1_score(yt, preds, zero_division=0)
    cm   = confusion_matrix(yt, preds)

    print(f"\n{'─'*56}")
    print(f"  {label}")
    print(f"{'─'*56}")
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def attack_success_rate(model, X_original, y_original, X_adversarial):
    """
    ASR = fraction of samples the model originally got RIGHT
          that it now gets WRONG after perturbation.
    """

    model.eval()
    with torch.no_grad():
        pred_orig = model.predict(
            torch.from_numpy(X_original).to(DEVICE)
        ).cpu().numpy()
        pred_adv = model.predict(
            torch.from_numpy(X_adversarial).to(DEVICE)
        ).cpu().numpy()

    correct_mask = pred_orig == y_original
    if correct_mask.sum() == 0:
        return 0.0
    return float((pred_adv[correct_mask] != y_original[correct_mask]).mean())


# =============================================================================
# 4. ATTACK IMPLEMENTATIONS
# =============================================================================

# --- 4a. DeepFool ------------------------------------------------------------

def deepfool_attack(model, X,
                    max_iter=50, overshoot=0.02,
                    epsilon=1e-6, batch_size=256, verbose=True):


    if verbose:
        print(f"\n[DeepFool] Generating {len(X)} AEs  "
              f"(max_iter={max_iter}, overshoot={overshoot})")

    model.eval()
    X_adv = X.copy()
    t0    = time.time()

    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(
            X[start: start + batch_size].copy()
        ).to(DEVICE)

        for _ in range(max_iter):
            xb_d = xb.detach().requires_grad_(True)
            out  = model(xb_d)                                   # (B,)
            grad = torch.autograd.grad(out.sum(), xb_d)[0]       # (B, F)

            with torch.no_grad():
                f_val     = out.detach()                         # (B,)
                w_norm    = grad.norm(dim=1).clamp(min=epsilon)  # (B,)
                dist      = (f_val / w_norm).unsqueeze(1)        # (B,1)
                direction = (
                    -torch.sign(f_val).unsqueeze(1)
                    * grad / w_norm.unsqueeze(1)
                )
                xb = xb + (dist + overshoot) * direction

        X_adv[start: start + batch_size] = xb.detach().cpu().numpy()

    if verbose:
        print(f"[DeepFool] Done in {time.time() - t0:.1f}s")

    return X_adv.astype(np.float32)


# --- 4b. ZOO -----------------------------------------------------------------

def zoo_attack(model, X, y,
               n_iter=20, lr=0.01, h=1e-4,
               crossover=0.7, max_samples=200, verbose=True):


    # Hard cap: ZOO is O(n * iters * features) - must limit on CPU
    if len(X) > max_samples:
        idx = np.random.choice(len(X), size=max_samples, replace=False)
        X   = X[idx]
        y   = y[idx]

    if verbose:
        print(f"\n[ZOO] Generating {len(X)} AEs  "
              f"(n_iter={n_iter}, crossover={crossover})")
        print(f"[ZOO] Estimated time on CPU: ~{len(X) * n_iter * 0.005:.0f}s")

    model.eval()
    X_adv     = X.copy().astype(np.float32)
    n, n_feat = X.shape
    t0        = time.time()

    with torch.no_grad():
        for i in range(n):
            x            = X_adv[i].copy()
            target_class = 1 - int(y[i])

            for _ in range(n_iter):
                mask = np.random.rand(n_feat) < crossover
                if not mask.any():
                    mask[np.random.randint(n_feat)] = True

                grad_est = np.zeros(n_feat, dtype=np.float32)
                for fi in np.where(mask)[0]:
                    xp, xm  = x.copy(), x.copy()
                    xp[fi] += h
                    xm[fi] -= h
                    fp = model(
                        torch.from_numpy(xp).unsqueeze(0).to(DEVICE)
                    ).item()
                    fm = model(
                        torch.from_numpy(xm).unsqueeze(0).to(DEVICE)
                    ).item()
                    grad_est[fi] = (fp - fm) / (2 * h)

                direction = -grad_est if target_class == 0 else grad_est
                x = x + lr * direction

            X_adv[i] = x

            if verbose and (i + 1) % 50 == 0:
                print(f"  [ZOO] {i+1}/{n} done  "
                      f"({time.time()-t0:.0f}s elapsed)")

    if verbose:
        print(f"[ZOO] Done in {time.time() - t0:.1f}s")

    return X_adv


# --- 4c. KDE -----------------------------------------------------------------

def kde_attack(X, y, ratio=0.05, bandwidth=1e-3,
               n_samples=None, verbose=True):
    """
    KDE Attack - fits a Gaussian KDE on the attack-class distribution and
    samples new adversarial examples from it.

    Paper reference: Section 2.5 and Table 7.
    Hyperparameters: bandwidth=0.00001 (scaled to 1e-3 for tabular stability).
    """

    if verbose:
        print(f"\n[KDE] Generating AEs  (bandwidth={bandwidth})")

    X_attack = X[y == 1]
    n_gen    = n_samples or max(1, int(len(X) * ratio))
    n_gen    = min(n_gen, len(X_attack) * 3)

    kde   = KernelDensity(kernel="gaussian", bandwidth=bandwidth, leaf_size=30)
    kde.fit(X_attack)
    X_adv = kde.sample(n_gen, random_state=SEED).astype(np.float32)

    if verbose:
        print(f"[KDE] Sampled {len(X_adv)} adversarial examples.")

    return X_adv


# --- 4d. GAN -----------------------------------------------------------------

class _Generator(nn.Module):
    """Latent noise  ->  synthetic attack-traffic features."""

    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class _Discriminator(nn.Module):
    """Real vs. generated attack-traffic classifier."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gan_attack(X, y, n_features,
               ratio=0.05, n_epochs=30, batch_size=128,
               latent_dim=100, lr=2e-4, verbose=True):
    """
    GAN Attack - trains a Generator/Discriminator pair on attack-class
    traffic, then generates synthetic adversarial examples.

    Paper reference: Section 2.3 and Table 5.
    Hyperparameters: Adam(beta1=0.5), BCELoss, dropout=0.4.
    """

    if verbose:
        print(f"\n[GAN] Training  (epochs={n_epochs}, latent_dim={latent_dim})")

    X_attack = torch.from_numpy(X[y == 1]).float().to(DEVICE)
    n_gen    = max(1, int(len(X) * ratio))

    G     = _Generator(latent_dim, n_features).to(DEVICE)
    D     = _Discriminator(n_features).to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce   = nn.BCELoss()

    loader = DataLoader(
        TensorDataset(X_attack),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )

    for epoch in range(1, n_epochs + 1):
        for (real_x,) in loader:
            b = real_x.size(0)

            # Discriminator step
            z      = torch.randn(b, latent_dim, device=DEVICE)
            fake   = G(z).detach()
            loss_D = (
                bce(D(real_x), torch.ones(b,  device=DEVICE)) +
                bce(D(fake),   torch.zeros(b, device=DEVICE))
            )
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Generator step
            z      = torch.randn(b, latent_dim, device=DEVICE)
            fake   = G(z)
            loss_G = bce(D(fake), torch.ones(b, device=DEVICE))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>3}/{n_epochs}  "
                  f"loss_D={loss_D.item():.4f}  "
                  f"loss_G={loss_G.item():.4f}")

    G.eval()
    with torch.no_grad():
        z     = torch.randn(n_gen, latent_dim, device=DEVICE)
        X_adv = G(z).cpu().numpy().astype(np.float32)

    if verbose:
        print(f"[GAN] Generated {len(X_adv)} adversarial examples.")

    return X_adv


# =============================================================================
# 5. ADVERSARIAL DATASET GENERATION PIPELINE
# =============================================================================

def generate_adversarial_dataset(model, X_train, y_train, n_features,
                                  ae_ratio=0.05, attacks=None, verbose=True):
    """
    Run each attack and append the resulting AEs to the original training set.

    Parameters
    ----------
    ae_ratio : float
        Fraction of training data converted to AEs per attack.
        Paper uses 5% (Table 12) and 25% (ablation) in defense experiments.
    attacks  : list[str]
        Any subset of ['deepfool', 'zoo', 'kde', 'gan'].

    Returns
    -------
    X_aug, y_aug : augmented dataset ready for retraining
    """

    attacks  = attacks or ["deepfool", "zoo", "kde", "gan"]
    n_ae     = max(1, int(len(X_train) * ae_ratio))
    idx_sub  = np.random.choice(len(X_train),
                                 size=min(n_ae, len(X_train)), replace=False)
    X_sub    = X_train[idx_sub]
    y_sub    = y_train[idx_sub]

    all_X, all_y = [], []

    if "deepfool" in attacks:
        Xa = deepfool_attack(model, X_sub, verbose=verbose)
        all_X.append(Xa); all_y.append(y_sub)

    if "zoo" in attacks:
        Xa = zoo_attack(model, X_sub, y_sub, verbose=verbose)
        all_X.append(Xa); all_y.append(y_sub)

    if "kde" in attacks:
        Xa = kde_attack(X_train, y_train, ratio=ae_ratio, verbose=verbose)
        all_X.append(Xa)
        all_y.append(np.ones(len(Xa), dtype=np.int64))

    if "gan" in attacks:
        Xa = gan_attack(X_train, y_train, n_features,
                        ratio=ae_ratio, verbose=verbose)
        all_X.append(Xa)
        all_y.append(np.ones(len(Xa), dtype=np.int64))

    X_aug = np.vstack([X_train] + all_X)
    y_aug = np.concatenate([y_train] + all_y)
    n_ae_total = sum(len(a) for a in all_X)

    if verbose:
        print(f"\n[Pipeline] Original: {len(X_train):,}  "
              f"AEs added: {n_ae_total:,}  "
              f"Augmented total: {len(X_aug):,}")

    return X_aug.astype(np.float32), y_aug.astype(np.int64)


# =============================================================================
# 6. EVALUATE MODEL UNDER ALL FOUR ATTACKS
# =============================================================================

def evaluate_against_all_attacks(model, X_test, y_test, n_features,
                                  attack_ratio=0.05,
                                  label_prefix="Model", verbose=True):
    """
    Inject adversarial examples into the test set at attack_ratio,
    then report accuracy and ASR for each attack type.
    """

    # Hard cap: DeepFool/ZOO iterate per-sample so must be limited on CPU.
    # 500 samples for DeepFool (~30s), 200 for ZOO (~2 min). KDE/GAN are fast.
    MAX_EVAL_SAMPLES = 500

    results = {}
    n_ae    = min(MAX_EVAL_SAMPLES, max(1, int(len(X_test) * attack_ratio)))
    idx_sub = np.random.choice(len(X_test), size=n_ae, replace=False)
    X_sub   = X_test[idx_sub]
    y_sub   = y_test[idx_sub]

    print(f"\n[Eval] Attack sample budget: {n_ae} "
          f"(capped at {MAX_EVAL_SAMPLES} for CPU speed)")

    attack_fns = {
        "DeepFool": lambda: deepfool_attack(model, X_sub, verbose=True),
        "ZOO"     : lambda: zoo_attack(model, X_sub, y_sub,
                                        max_samples=200, verbose=True),
        "KDE"     : lambda: kde_attack(X_test, y_test,
                                        ratio=attack_ratio, verbose=True),
        "GAN"     : lambda: gan_attack(X_test, y_test, n_features,
                                        ratio=attack_ratio,
                                        n_epochs=15, verbose=True),
    }

    for name, fn in attack_fns.items():
        X_adv = fn()

        rest_mask = np.ones(len(X_test), bool)
        rest_mask[idx_sub] = False

        if name in ("KDE", "GAN"):
            X_eval = np.vstack([X_test[rest_mask], X_adv])
            y_eval = np.concatenate(
                [y_test[rest_mask], np.ones(len(X_adv), dtype=np.int64)]
            )
            n      = min(len(X_sub), len(X_adv))
            asr    = attack_success_rate(
                model,
                X_adv[:n],
                np.ones(n, dtype=np.int64),
                X_adv[:n],
            )
        else:
            X_eval = np.vstack([X_test[rest_mask], X_adv])
            y_eval = np.concatenate([y_test[rest_mask], y_sub])
            asr    = attack_success_rate(model, X_sub, y_sub, X_adv)

        lbl     = (f"{label_prefix} | under {name} "
                   f"({attack_ratio*100:.0f}% AE injection)")
        metrics = evaluate_model(model, X_eval, y_eval, label=lbl)
        metrics["asr"] = asr
        print(f"  Attack Success Rate (ASR): {asr:.4f}  ({asr*100:.2f}%)")
        results[name] = metrics

    return results


# =============================================================================
# 7. DEFENSE: ADVERSARIAL TRAINING  (paper Section 4.2)
# =============================================================================

def build_defense_model(X_train, y_train, n_features,
                         ae_ratio=0.05, defense_attacks=None,
                         epochs=30, verbose=True):
 

    defense_attacks = defense_attacks or ["deepfool"]

    print(f"\n{'='*60}")
    print("  PHASE 1: Train Clean Baseline")
    print(f"{'='*60}")
    baseline = IDSClassifier(n_features).to(DEVICE)
    train_model(baseline, X_train, y_train, epochs=epochs, verbose=verbose)

    print(f"\n{'='*60}")
    print(f"  PHASE 2: Generate AEs  |  attacks={defense_attacks}"
          f"  |  ratio={ae_ratio*100:.0f}%")
    print(f"{'='*60}")
    X_aug, y_aug = generate_adversarial_dataset(
        model=baseline,
        X_train=X_train,
        y_train=y_train,
        n_features=n_features,
        ae_ratio=ae_ratio,
        attacks=defense_attacks,
        verbose=verbose,
    )

    print(f"\n{'='*60}")
    print("  PHASE 3: Retrain on Augmented Dataset")
    print(f"{'='*60}")
    defense_model = IDSClassifier(n_features).to(DEVICE)
    train_model(defense_model, X_aug, y_aug, epochs=epochs, verbose=verbose)

    return defense_model


# =============================================================================
# 8. RESULTS SUMMARY TABLE
# =============================================================================

def print_comparison_table(baseline_results, defense_results, attack_ratio):
    """
    Side-by-side table of Baseline vs. Defense accuracy and ASR,
    matching the style of Tables 12-14 in the paper.
    """

    w = 74
    print(f"\n{'='*w}")
    print(f"  RESULTS SUMMARY  (attack injection ratio = {attack_ratio*100:.0f}%)")
    print(f"{'='*w}")
    print(
        f"  {'Attack':<10} {'Baseline Acc':>13} {'Defense Acc':>13} "
        f"{'Delta Acc':>10} {'Base ASR':>10} {'Def ASR':>10}"
    )
    print(f"{'─'*w}")

    for atk in ["DeepFool", "ZOO", "KDE", "GAN"]:
        b     = baseline_results.get(atk, {})
        d     = defense_results.get(atk, {})
        b_acc = b.get("accuracy", float("nan"))
        d_acc = d.get("accuracy", float("nan"))
        b_asr = b.get("asr",      float("nan"))
        d_asr = d.get("asr",      float("nan"))
        delta = d_acc - b_acc
        print(
            f"  {atk:<10} {b_acc:>13.4f} {d_acc:>13.4f} "
            f"{delta:>+10.4f} {b_asr:>10.4f} {d_asr:>10.4f}"
        )

    print(f"{'='*w}")
    print("  Delta Acc > 0  =>  defense IMPROVED accuracy under that attack.")
    print(f"{'='*w}\n")


# =============================================================================
# 9. ARGUMENT PARSER
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Adversarial Defense for CICIDS2018 IDS"
    )
    p.add_argument(
        "--data", type=str, default=DATASET_PATH,
        help="CSV path (defaults to the hardcoded DATASET_PATH above).",
    )
    p.add_argument(
        "--mode", choices=["fast", "full"], default="fast",
        help=(
            "fast = DeepFool-only defense (paper default). "
            "full = all four attacks used in defense training."
        ),
    )
    p.add_argument(
        "--ae_ratio", type=float, default=0.05,
        help="Fraction of training data injected as AEs  (default 0.05 = 5%%)",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs  (default 30, matches paper)",
    )
    p.add_argument(
        "--attack_ratio", type=float, default=0.05,
        help="Fraction of test data replaced with AEs during eval  (default 0.05)",
    )
    return p.parse_args()


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("  Unified Adversarial Defense System for IDS  (CICIDS2018)")
    print(f"  Device       : {DEVICE}")
    print(f"  Mode         : {args.mode.upper()}")
    print(f"  Dataset      : {args.data}")
    print(f"  AE ratio     : {args.ae_ratio*100:.0f}%  (training injection)")
    print(f"  Attack ratio : {args.attack_ratio*100:.0f}%  (evaluation injection)")
    print("=" * 70)

    # 1. Load data
    X_train, X_test, y_train, y_test, n_features = load_cicids2018(args.data)

    # 2. Train and evaluate baseline on clean data
    print(f"\n{'='*60}")
    print("  BASELINE MODEL  (clean training data only)")
    print(f"{'='*60}")
    baseline = IDSClassifier(n_features).to(DEVICE)
    train_model(baseline, X_train, y_train, epochs=args.epochs)
    base_clean = evaluate_model(baseline, X_test, y_test,
                                 label="Baseline | clean test set")

    # 3. Attack the baseline
    print(f"\n{'='*60}")
    print("  ATTACKING THE BASELINE MODEL")
    print(f"{'='*60}")
    baseline_results = evaluate_against_all_attacks(
        model=baseline,
        X_test=X_test,
        y_test=y_test,
        n_features=n_features,
        attack_ratio=args.attack_ratio,
        label_prefix="Baseline",
    )

    # 4. Build the defense model
    defense_attacks = (
        ["deepfool"] if args.mode == "fast"
        else ["deepfool", "zoo", "kde", "gan"]
    )
    defense_model = build_defense_model(
        X_train=X_train,
        y_train=y_train,
        n_features=n_features,
        ae_ratio=args.ae_ratio,
        defense_attacks=defense_attacks,
        epochs=args.epochs,
    )

    # 5. Attack the defense model
    print(f"\n{'='*60}")
    print("  ATTACKING THE DEFENSE MODEL")
    print(f"{'='*60}")
    def_clean = evaluate_model(defense_model, X_test, y_test,
                                label="Defense | clean test set")
    defense_results = evaluate_against_all_attacks(
        model=defense_model,
        X_test=X_test,
        y_test=y_test,
        n_features=n_features,
        attack_ratio=args.attack_ratio,
        label_prefix="Defense",
    )

    # 6. Print summary
    print_comparison_table(baseline_results, defense_results, args.attack_ratio)
    print("  Clean-set accuracy")
    print(f"  Baseline : {base_clean['accuracy']:.4f}  "
          f"Defense  : {def_clean['accuracy']:.4f}  "
          f"Delta = {def_clean['accuracy'] - base_clean['accuracy']:+.4f}")
    print()


if __name__ == "__main__":
    main()