import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


# ── LOAD DATA ──────────────────────────────────────────────────────────────────

df = pd.read_csv("data\\cic1.csv", low_memory=False)

possible_labels = ["Label","label","Class","class","Attack","attack"]
label_col = None
for col in df.columns:
    if col in possible_labels:
        label_col = col
        break

if label_col is None:
    raise Exception("No label column found.")

print("Label column detected:", label_col)

y = df[label_col]
X = df.drop(columns=[label_col])
X = X.select_dtypes(include=[np.number])
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]
X = X[(np.abs(X) < 1e10).all(axis=1)]
y = y.loc[X.index]

print("Clean rows:", len(X))

# ── SAMPLE ─────────────────────────────────────────────────────────────────────

MAX_SAMPLES = 50000
if len(X) > MAX_SAMPLES:
    idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
    X = X.iloc[idx]
    y = y.iloc[idx]

print("Samples:", len(X))

# ── ENCODE & SCALE ─────────────────────────────────────────────────────────────

le = LabelEncoder()
y  = le.fit_transform(y)
num_classes = len(np.unique(y))
print("Classes:", num_classes)

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── SPLIT ──────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

# ── MODEL DEFINITION ───────────────────────────────────────────────────────────

class IDS(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  DEEPFOOL
# ══════════════════════════════════════════════════════════════════════════════

def deepfool(x, model, num_classes, overshoot=0.02, max_iter=50):
    x0    = x.clone().detach()
    x     = x.clone().detach().requires_grad_(True)
    label = model(x).argmax().item()
    pert  = torch.zeros_like(x)

    for _ in range(max_iter):
        logits = model(x)
        if logits.argmax().item() != label:
            return x.detach(), True
        grads = []
        for k in range(num_classes):
            model.zero_grad()
            logits[0, k].backward(retain_graph=True)
            grads.append(x.grad.clone())
            x.grad.zero_()
        grads = torch.stack(grads)
        w     = grads - grads[label]
        f     = logits.detach()[0] - logits.detach()[0, label]
        norms = w.view(num_classes, -1).norm(dim=1) + 1e-8
        dist  = torch.abs(f) / norms
        dist[label] = 1e9
        idx   = dist.argmin()
        r     = (dist[idx] * w[idx] / norms[idx]).reshape_as(x)
        pert += r
        x     = (x0 + (1 + overshoot) * pert).detach().requires_grad_(True)

    return x.detach(), False


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — TRAIN BASELINE MODEL (no defense)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  STEP 1: Training baseline model (no defense)...")
print("="*62)

baseline_model = IDS(X_train.shape[1], num_classes)
opt     = optim.Adam(baseline_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(15):
    baseline_model.train()
    out  = baseline_model(X_train)
    loss = loss_fn(out, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"  Epoch {epoch+1:>2} Loss: {loss.item():.4f}")

baseline_model.eval()
with torch.no_grad():
    baseline_preds = baseline_model(X_test).argmax(1)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"\n  Baseline Model Accuracy (clean): {baseline_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — GENERATE ADVERSARIAL EXAMPLES FROM BASELINE MODEL
#  We generate adversarial examples using 5% of the training set.
#  These will be mixed into the training data for the defense model.
#  Why 5%? The paper (Ahmed et al.) shows 5% AEs gives best balance
#  between robustness improvement and clean accuracy retention.
# ══════════════════════════════════════════════════════════════════════════════

ADV_RATIO   = 0.05   # 5% of training set
N_ADV       = int(len(X_train) * ADV_RATIO)

print("\n" + "="*62)
print(f"  STEP 2: Generating {N_ADV} adversarial examples ({int(ADV_RATIO*100)}% of train set)...")
print("="*62)

adv_samples = []
adv_labels  = []

baseline_model.eval()
generated = 0

for i in range(len(X_train)):
    if generated >= N_ADV:
        break
    adv, fooled = deepfool(X_train[i:i+1], baseline_model, num_classes)
    if fooled:
        adv_samples.append(adv)
        adv_labels.append(y_train[i].item())   # keep ORIGINAL true label
        generated += 1
        if generated % 200 == 0:
            print(f"  Generated {generated}/{N_ADV} adversarial examples...")

adv_X = torch.cat(adv_samples, dim=0)
adv_y = torch.tensor(adv_labels, dtype=torch.long)

print(f"  Done. Generated {len(adv_X)} adversarial examples.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — BUILD AUGMENTED TRAINING SET
#  Mix original training data + adversarial examples.
#  The model learns: "even if input is perturbed this way, still predict
#  the correct original class."
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  STEP 3: Building augmented training set...")
print("="*62)

X_train_aug = torch.cat([X_train, adv_X], dim=0)
y_train_aug = torch.cat([y_train, adv_y], dim=0)

# Shuffle the augmented set so adversarial examples aren't all at the end
perm        = torch.randperm(len(X_train_aug))
X_train_aug = X_train_aug[perm]
y_train_aug = y_train_aug[perm]

print(f"  Original training samples : {len(X_train)}")
print(f"  Adversarial examples added: {len(adv_X)}")
print(f"  Total augmented samples   : {len(X_train_aug)}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN DEFENSE MODEL ON AUGMENTED DATA
#  Same architecture as baseline — the robustness comes from the data,
#  not from changing the model structure.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  STEP 4: Training defense model on augmented data...")
print("="*62)

defense_model = IDS(X_train.shape[1], num_classes)
opt_d   = optim.Adam(defense_model.parameters(), lr=0.001)

for epoch in range(15):
    defense_model.train()
    out  = defense_model(X_train_aug)
    loss = loss_fn(out, y_train_aug)
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()
    print(f"  Epoch {epoch+1:>2} Loss: {loss.item():.4f}")

defense_model.eval()
with torch.no_grad():
    defense_preds = defense_model(X_test).argmax(1)
defense_clean_acc = accuracy_score(y_test, defense_preds)
print(f"\n  Defense Model Accuracy (clean): {defense_clean_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — EVALUATE BOTH MODELS UNDER DEEPFOOL ATTACK
#  Attack 1%–5% of test samples and compare accuracy of
#  baseline model vs defense model side by side.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  STEP 5: Evaluating both models under DeepFool attack...")
print("="*62)

PERCENTAGES    = [1, 2, 3, 4, 5]
TOTAL_SAMPLES  = len(X_test)

baseline_model.eval()
defense_model.eval()

with torch.no_grad():
    all_clean_base    = baseline_model(X_test).argmax(1).numpy()
    all_clean_defense = defense_model(X_test).argmax(1).numpy()

sweep = []

for pct in PERCENTAGES:
    n_attack = max(1, int(TOTAL_SAMPLES * pct / 100))
    print(f"\n  Attacking {pct}% of test samples ({n_attack} samples)...")

    base_preds    = all_clean_base.copy()
    defense_preds = all_clean_defense.copy()

    for i in range(n_attack):

        # Attack baseline model
        adv_b, _ = deepfool(X_test[i:i+1], baseline_model, num_classes)
        with torch.no_grad():
            base_preds[i] = baseline_model(adv_b).argmax().item()

        # Attack defense model with the SAME adversarial sample
        # (generated from baseline — simulates black-box transfer attack)
        with torch.no_grad():
            defense_preds[i] = defense_model(adv_b).argmax().item()

    acc_base    = accuracy_score(y_test.numpy(), base_preds)
    acc_defense = accuracy_score(y_test.numpy(), defense_preds)

    sweep.append({
        "pct"        : pct,
        "base_acc"   : acc_base,
        "defense_acc": acc_defense,
        "improvement": acc_defense - acc_base,
    })

    print(f"  Baseline accuracy    : {acc_base:.4f}")
    print(f"  Defense  accuracy    : {acc_defense:.4f}")
    print(f"  Improvement          : +{acc_defense - acc_base:.4f}")


# ── FINAL RESULTS TABLE ────────────────────────────────────────────────────────

print("\n\n" + "="*72)
print("         DEFENSE MODEL RESULTS  --  BASELINE vs ADVERSARIAL TRAINING")
print("="*72)
print(f"  Clean accuracy  |  Baseline : {baseline_acc:.4f}  |  Defense : {defense_clean_acc:.4f}")
print("-"*72)
print(f"  {'% Attacked':>10} | {'Baseline Acc':>14} | {'Defense Acc':>13} | {'Improvement':>12}")
print("-"*72)

for r in sweep:
    print(f"  {r['pct']:>9}%  | "
          f"  {r['base_acc']:>11.4f}  | "
          f"  {r['defense_acc']:>10.4f}  | "
          f"  +{r['improvement']:>9.4f}")

print("="*72)
print("\n  Defense strategy : Adversarial training with 5% DeepFool-generated AEs")
print("  Attack type      : DeepFool white-box transfer attack")
print("  Dataset          : CIC (subsampled 50,000 rows, 3 classes)")
print("="*72)