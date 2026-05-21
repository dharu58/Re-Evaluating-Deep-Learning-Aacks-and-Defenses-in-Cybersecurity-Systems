import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ── LOAD DATA ──────────────────────────────────────────────────────────────────

df = pd.read_csv("data\\cic1.csv", low_memory=False)

possible_labels = ["Label","label","Class","class","Attack","attack"]
label_col = None
for col in df.columns:
    if col in possible_labels:
        label_col = col
        break

if label_col is None:
    raise Exception("No label column found. Columns:\n" + str(df.columns))

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
y = le.fit_transform(y)
num_classes = len(np.unique(y))
print("Classes:", num_classes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── SPLIT ──────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
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

model.eval()
with torch.no_grad():
    baseline_preds = model(X_test).argmax(1)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"\nBaseline Accuracy: {baseline_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION MODULE
#  Strategy: three detectors run in parallel on every incoming sample.
#  A sample is flagged as adversarial if ANY detector triggers.
#
#  Detector 1 — Confidence Drop
#    Clean samples that the model predicts correctly tend to have high softmax
#    confidence. DeepFool pushes samples just past the decision boundary, so
#    the winning class often has LOW confidence (close to the runner-up).
#    If max softmax probability < threshold → flag.
#
#  Detector 2 — Prediction Instability (local smoothing check)
#    Add tiny Gaussian noise to the sample 10 times and re-predict.
#    A clean sample sitting firmly in one region will predict the same class
#    every time. An adversarial sample sitting right on the boundary will
#    flip its prediction under tiny noise.
#    If prediction changes more than K times out of 10 → flag.
#
#  Detector 3 — Input Perturbation Magnitude
#    Compare the incoming sample to its nearest clean training neighbour.
#    If the L2 distance is unusually large relative to the training distribution
#    → flag. DeepFool adds a small but detectable perturbation vector.
#    We approximate this with per-feature z-score — if any feature value
#    is an extreme outlier beyond what the training set contains → flag.
# ══════════════════════════════════════════════════════════════════════════════

import torch.nn.functional as F

# ── Pre-compute training set statistics for Detector 3 ────────────────────────
train_mean = X_train.mean(dim=0)
train_std  = X_train.std(dim=0) + 1e-8   # avoid division by zero

# Thresholds (tunable)
CONFIDENCE_THRESHOLD   = 0.70   # below this → suspicious
INSTABILITY_THRESHOLD  = 3      # if prediction flips ≥ 3/10 times → suspicious
ZSCORE_THRESHOLD       = 5.0    # if any feature is >5 std devs from mean → suspicious
NOISE_SCALE            = 0.02   # std of Gaussian noise for Detector 2
N_NOISE_TRIALS         = 10     # how many noisy copies to test


def detector_confidence(x, model):
    """
    Detector 1: Softmax confidence check.
    DeepFool places samples right at the decision boundary → low confidence.
    Returns True (flagged) if max probability is below threshold.
    """
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        max_prob = probs.max().item()
    flagged = max_prob < CONFIDENCE_THRESHOLD
    return flagged, max_prob


def detector_instability(x, model):
    """
    Detector 2: Prediction instability under noise.
    Clean samples sit firmly in a class region → stable predictions.
    Adversarial samples sit on the boundary → unstable predictions.
    Returns True (flagged) if prediction flips too many times.
    """
    with torch.no_grad():
        original_pred = model(x).argmax(1).item()
        flip_count = 0
        for _ in range(N_NOISE_TRIALS):
            noise      = torch.randn_like(x) * NOISE_SCALE
            noisy_pred = model(x + noise).argmax(1).item()
            if noisy_pred != original_pred:
                flip_count += 1
    flagged = flip_count >= INSTABILITY_THRESHOLD
    return flagged, flip_count


def detector_zscore(x):
    """
    Detector 3: Feature z-score outlier check.
    Computes how many standard deviations each feature is from the training mean.
    DeepFool adds a perturbation vector — this shifts some features into
    unusual territory relative to the training distribution.
    Returns True (flagged) if any feature exceeds the z-score threshold.
    """
    z_scores   = torch.abs((x - train_mean) / train_std)
    max_zscore = z_scores.max().item()
    flagged    = max_zscore > ZSCORE_THRESHOLD
    return flagged, max_zscore


def runtime_detector(x, model):
    """
    Master detector: runs all 3 detectors and flags if ANY triggers.
    Returns a verdict dict with full diagnostic info.
    """
    flag1, conf      = detector_confidence(x, model)
    flag2, flips     = detector_instability(x, model)
    flag3, max_z     = detector_zscore(x)

    is_adversarial   = flag1 or flag2 or flag3

    return {
        "adversarial"        : is_adversarial,
        "confidence_flag"    : flag1,
        "instability_flag"   : flag2,
        "zscore_flag"        : flag3,
        "max_confidence"     : round(conf,   4),
        "prediction_flips"   : flips,
        "max_zscore"         : round(max_z,  4),
    }


# ── DEEPFOOL (same as before) ──────────────────────────────────────────────────

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


# ── RUN: ATTACK + DETECT ───────────────────────────────────────────────────────

SAMPLES = min(200, len(X_test))
print(f"\nRunning DeepFool attack + runtime detection on {SAMPLES} samples...")
print("(Each sample is attacked, then passed through the detector)\n")

results = []

for i in range(SAMPLES):
    original = X_test[i:i+1]

    # Step 1 — generate adversarial sample
    adv, fooled = deepfool(original, model, num_classes)

    # Step 2 — run detector on the adversarial sample
    verdict = runtime_detector(adv, model)

    # Step 3 — also run detector on the clean sample (for comparison)
    clean_verdict = runtime_detector(original, model)

    results.append({
        "sample_idx"          : i,
        "attack_succeeded"    : fooled,
        "detected_as_adv"     : verdict["adversarial"],
        "clean_flagged"       : clean_verdict["adversarial"],   # false positive check
        "confidence_flag"     : verdict["confidence_flag"],
        "instability_flag"    : verdict["instability_flag"],
        "zscore_flag"         : verdict["zscore_flag"],
        "max_confidence"      : verdict["max_confidence"],
        "prediction_flips"    : verdict["prediction_flips"],
        "max_zscore"          : verdict["max_zscore"],
    })

# ── METRICS ────────────────────────────────────────────────────────────────────

total            = len(results)
attacked         = sum(r["attack_succeeded"]  for r in results)
detected         = sum(r["detected_as_adv"] and r["attack_succeeded"] for r in results)
missed           = attacked - detected
false_positives  = sum(r["clean_flagged"]     for r in results)

detection_rate   = detected / attacked       if attacked       > 0 else 0
false_pos_rate   = false_positives / total

# Which detector triggered most
conf_triggers    = sum(r["confidence_flag"]   for r in results)
inst_triggers    = sum(r["instability_flag"]  for r in results)
zscore_triggers  = sum(r["zscore_flag"]       for r in results)

# ── PRINT RESULTS ──────────────────────────────────────────────────────────────

print("=" * 62)
print("       DEEPFOOL DETECTION  —  RUNTIME DETECTOR RESULTS")
print("=" * 62)
print(f"  Total samples tested          : {total}")
print(f"  Successfully attacked         : {attacked}")
print(f"  Detected as adversarial       : {detected}")
print(f"  Missed (not detected)         : {missed}")
print(f"  False positives (clean→flagged): {false_positives}")
print("-" * 62)
print(f"  Detection Rate                : {detection_rate*100:.2f}%")
print(f"  False Positive Rate           : {false_pos_rate*100:.2f}%")
print("-" * 62)
print(f"  Detector 1 (confidence) triggered : {conf_triggers} times")
print(f"  Detector 2 (instability) triggered: {inst_triggers} times")
print(f"  Detector 3 (z-score) triggered    : {zscore_triggers} times")
print("=" * 62)

# ── SAMPLE-LEVEL ALERT LOG (first 20) ─────────────────────────────────────────

print("\nSample-level alert log (first 20 samples):")
print(f"  {'Idx':>4} | {'Attacked':>8} | {'Detected':>8} | {'Conf':>6} | {'Flips':>5} | {'Z-score':>7} | Alert")
print("-" * 70)

for r in results[:20]:
    alert = "⚠ ADVERSARIAL" if r["detected_as_adv"] else "  clean"
    print(f"  {r['sample_idx']:>4} | "
          f"{'yes' if r['attack_succeeded'] else 'no':>8} | "
          f"{'yes' if r['detected_as_adv'] else 'no':>8} | "
          f"{r['max_confidence']:>6.3f} | "
          f"{r['prediction_flips']:>5} | "
          f"{r['max_zscore']:>7.2f} | "
          f"{alert}")

print("=" * 62)