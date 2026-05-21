import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    raise Exception("No label column found. Columns are:\n" + str(df.columns))

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

# ── ENCODE & NORMALIZE ─────────────────────────────────────────────────────────

le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))
print("Classes:", num_classes)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ── SPLIT ──────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

# ── MODEL ──────────────────────────────────────────────────────────────────────

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

model = IDS(X_train.shape[1], num_classes)

# ── TRAIN ──────────────────────────────────────────────────────────────────────

print("\nTraining model...")

opt     = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(15):
    model.train()
    out  = model(X_train)
    loss = loss_fn(out, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# ── BASELINE ACCURACY ──────────────────────────────────────────────────────────

model.eval()

with torch.no_grad():
    preds = model(X_test).argmax(1)

baseline_acc = accuracy_score(y_test, preds)
print(f"\nBaseline Accuracy: {baseline_acc:.4f}")

# ── DEEPFOOL ATTACK ────────────────────────────────────────────────────────────

def deepfool(x, model, num_classes, overshoot=0.02, max_iter=50):
    x0 = x.clone().detach()
    x  = x.clone().detach().requires_grad_(True)

    logits = model(x)
    label  = logits.argmax().item()
    pert   = torch.zeros_like(x)

    for _ in range(max_iter):
        logits = model(x)
        pred   = logits.argmax().item()

        if pred != label:
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
        pert  += r
        x     = (x0 + (1 + overshoot) * pert).detach().requires_grad_(True)

    return x.detach(), False

# ── PERCENTAGE SWEEP (attack 1%–5% of test samples, rest stay clean) ──────────

TOTAL_SAMPLES = len(X_test)
PERCENTAGES   = [1, 2, 3, 4, 5]   # % of test set to attack

sweep_results = []

# get clean predictions for all test samples (baseline)
model.eval()
with torch.no_grad():
    all_clean_preds = model(X_test).argmax(1).numpy()

for pct in PERCENTAGES:
    n_attack = max(1, int(TOTAL_SAMPLES * pct / 100))

    print(f"\nRunning DeepFool on {pct}% of test samples ({n_attack} samples)...")

    # start from clean predictions, overwrite attacked ones
    final_preds = all_clean_preds.copy()

    for i in range(n_attack):
        adv, _ = deepfool(X_test[i:i+1], model, num_classes, overshoot=0.02)
        with torch.no_grad():
            pred = model(adv).argmax().item()
        final_preds[i] = pred

    acc_after = accuracy_score(y_test.numpy(), final_preds)
    drop      = baseline_acc - acc_after

    sweep_results.append({
        "pct"      : pct,
        "acc_after": acc_after,
        "drop"     : drop,
    })

# ── PRINT FINAL RESULTS TABLE ─────────────────────────────────────────────────

print("\n")
print("=" * 62)
print("          DEEPFOOL ATTACK  —  IDS MODEL  —  RESULTS TABLE")
print("=" * 62)
print(f"  Baseline Accuracy (before any attack) : {baseline_acc:.4f}")
print("-" * 62)
print(f"  {'% Attacked':>12} | {'Acc After Attack':>16} | {'Acc Drop':>10}")
print("-" * 62)

for r in sweep_results:
    print(f"  {r['pct']:>11}%  | "
          f"  {r['acc_after']:>13.4f}   | "
          f"  -{r['drop']:>7.4f}  ")

print("=" * 62)