# ============================================================
# 🎓 ISTQB CT-AI — Complete Beginner Project (Google Colab)
# JK Tech Learning Lab | Author: Jag
# Covers ALL syllabus chapters in one runnable program
# ============================================================
# Chapters Covered:
#  Ch1 - Introduction to AI & Testing
#  Ch2 - AI Techniques (ML, Deep Learning, NLP)
#  Ch3 - ML Overview (Supervised, Unsupervised, Reinforcement)
#  Ch4 - ML Data (Quality, Bias, Preprocessing)
#  Ch5 - Performance Metrics (Classification & Regression)
#  Ch6 - Deep Learning & CNNs
#  Ch7 - Testing AI Systems (Specific Challenges & Strategies)
# ============================================================

# ── INSTALL (only needed first time in Colab) ──────────────
# !pip install scikit-learn pandas numpy matplotlib seaborn tensorflow --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris, make_regression, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve
)

print("=" * 60)
print("🎓  ISTQB CT-AI  ·  Complete Beginner Project")
print("    JK Tech Learning Lab")
print("=" * 60)


# ============================================================
# CHAPTER 1 — Introduction to AI & Testing
# ============================================================
print("\n" + "─" * 60)
print("📘 CHAPTER 1 — Introduction to AI & Testing")
print("─" * 60)

print("""
AI DEFINITION:
  Artificial Intelligence = systems that mimic human
  cognitive functions (learning, reasoning, problem-solving).

KEY AI BRANCHES:
  • Machine Learning    — learns from data (no explicit rules)
  • Deep Learning       — neural networks with many layers
  • NLP                 — understanding/generating human language
  • Computer Vision     — interpreting images/video
  • Expert Systems      — rule-based knowledge bases

WHY TESTING AI IS DIFFERENT:
  • No deterministic output — same input can give different results
  • Data-dependent behaviour — garbage in = garbage out
  • Model drift — performance degrades over time
  • Bias & fairness — model may discriminate
  • Explainability — black-box models are hard to verify

ISTQB AI TESTING ROLES:
  Tester → validates data quality, model metrics, fairness,
           bias, robustness, and explainability.
""")


# ============================================================
# CHAPTER 2 — AI Techniques
# ============================================================
print("─" * 60)
print("📘 CHAPTER 2 — AI Techniques Overview")
print("─" * 60)

print("""
MACHINE LEARNING TYPES (Preview — deep dive in Ch3):
  1. Supervised    → labelled data  (classification, regression)
  2. Unsupervised  → no labels      (clustering, anomaly detection)
  3. Reinforcement → reward signal  (game agents, robotics)

NLP CONCEPTS:
  • Tokenisation  — splitting text into tokens/words
  • Stemming      — reducing words to root form
  • Sentiment     — positive / negative / neutral

NEURAL NETWORKS:
  Input Layer → Hidden Layers → Output Layer
  Each layer learns increasingly abstract features.

EXPERT SYSTEMS:
  IF temperature > 38 AND cough == True THEN flag_flu = True
  Rule-based; predictable; easy to test; brittle.
""")

# Simple NLP demonstration — tokenisation & sentiment
text_samples = [
    "The AI model works perfectly and is very accurate.",
    "This system is terrible and produces wrong results.",
    "The performance is average, nothing special."
]

def simple_sentiment(text):
    positive_words = {"perfectly", "accurate", "great", "excellent", "good", "works"}
    negative_words = {"terrible", "wrong", "bad", "poor", "fails", "broken"}
    tokens = text.lower().split()
    pos = sum(1 for w in tokens if w.strip(".,!") in positive_words)
    neg = sum(1 for w in tokens if w.strip(".,!") in negative_words)
    if pos > neg:   return "POSITIVE 😊"
    elif neg > pos: return "NEGATIVE 😟"
    else:           return "NEUTRAL 😐"

print("NLP Demo — Simple Sentiment Analysis:")
for t in text_samples:
    print(f"  [{simple_sentiment(t)}] {t[:55]}...")


# ============================================================
# CHAPTER 3 — ML Overview
# ============================================================
print("\n" + "─" * 60)
print("📘 CHAPTER 3 — ML Overview (Supervised | Unsupervised | RL)")
print("─" * 60)

# ── SUPERVISED: Classification (Iris dataset) ──────────────
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=50, random_state=42),
}

print("\n📊 SUPERVISED LEARNING — Iris Classification Comparison")
print(f"  {'Model':<22} {'Accuracy':>9} {'CV Mean':>9} {'CV Std':>8}")
print(f"  {'-'*50}")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    cv  = cross_val_score(model, X_train_s, y_train, cv=5)
    print(f"  {name:<22} {acc:>8.1%} {cv.mean():>8.1%} ±{cv.std():>5.3f}")

# ── UNSUPERVISED: Clustering ────────────────────────────────
print("\n📊 UNSUPERVISED LEARNING — KMeans Clustering")
X_blob, _ = make_blobs(n_samples=150, centers=3, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_blob)
unique, counts = np.unique(labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} samples")

print("""
📊 REINFORCEMENT LEARNING — Concept Only (no live agent here)
  Agent learns by trial & error using reward signals.
  Example: AlphaGo, autonomous driving, chatbot tuning (RLHF).
  Testing RL: check reward convergence, policy stability,
  edge-case behaviour, and safety constraints.
""")


# ============================================================
# CHAPTER 4 — ML Data Quality & Bias
# ============================================================
print("─" * 60)
print("📘 CHAPTER 4 — ML Data: Quality, Bias & Preprocessing")
print("─" * 60)

# Build a deliberately messy dataset
raw_data = pd.DataFrame({
    "age":    [25, np.nan, 35, 22, 45, 200, 30, np.nan, 28, 33],
    "salary": [50000, 60000, np.nan, 45000, 80000, 90000, 55000, 62000, np.nan, 70000],
    "gender": ["M","F","M","F","M","M","F",None,"F","M"],
    "score":  [85, 90, 78, 92, 88, 95, 82, 79, 91, 87],
})

print("\n🔍 DATA QUALITY CHECKS")
print(f"  Shape            : {raw_data.shape}")
print(f"  Missing values   :\n{raw_data.isnull().sum().to_string()}")
print(f"\n  Age outliers (>100): {(raw_data['age'] > 100).sum()} found")

# Fix issues
clean = raw_data.copy()
clean.loc[clean["age"] > 100, "age"] = np.nan          # remove outlier
clean["age"]    = clean["age"].fillna(clean["age"].median())
clean["salary"] = clean["salary"].fillna(clean["salary"].median())
clean["gender"] = clean["gender"].fillna("Unknown")

print(f"\n✅ After Cleaning — Missing values:\n{clean.isnull().sum().to_string()}")

# Bias check
print("\n⚖️  BIAS CHECK — Average score by gender:")
bias_check = clean.groupby("gender")["score"].mean()
print(bias_check.to_string())
print("""
  ISTQB Bias Types to Know:
  • Sampling bias    — data not representative of real population
  • Label bias       — human annotators introduce subjective error
  • Confirmation bias— model trained to confirm existing beliefs
  • Measurement bias — sensor/collection error skews data
  Testing for bias: check per-group metrics (accuracy, recall)
  and flag groups where performance deviates significantly.
""")

# Feature Engineering
le = LabelEncoder()
clean["gender_enc"] = le.fit_transform(clean["gender"])
print("✅ Feature Engineering — gender encoded:", dict(zip(le.classes_, le.transform(le.classes_))))


# ============================================================
# CHAPTER 5 — Performance Metrics
# ============================================================
print("\n" + "─" * 60)
print("📘 CHAPTER 5 — Performance Metrics")
print("─" * 60)

# ── Classification Metrics ──────────────────────────────────
best_model = RandomForestClassifier(n_estimators=50, random_state=42)
best_model.fit(X_train_s, y_train)
y_pred = best_model.predict(X_test_s)

print("\n📊 CLASSIFICATION METRICS (Random Forest on Iris)")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("""
  METRIC DEFINITIONS:
  Accuracy  = (TP+TN)/(TP+TN+FP+FN)  — overall correctness
  Precision = TP/(TP+FP)              — of all positives predicted, how many correct?
  Recall    = TP/(TP+FN)              — of all actual positives, how many found?
  F1        = 2*(P*R)/(P+R)           — harmonic mean; use when P & R both matter
  AUC-ROC   — area under ROC curve; 1.0 = perfect; 0.5 = random
""")

print("📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("🔢 Confusion Matrix:")
cm_df = pd.DataFrame(cm,
    index   =[f"True {n}"  for n in iris.target_names],
    columns =[f"Pred {n}"  for n in iris.target_names]
)
print(cm_df.to_string())

# ── Regression Metrics ──────────────────────────────────────
print("\n📊 REGRESSION METRICS")
X_r, y_r = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
reg = LinearRegression().fit(Xr_tr, yr_tr)
yr_pred = reg.predict(Xr_te)
print(f"  MAE  : {mean_absolute_error(yr_te, yr_pred):.4f}")
print(f"  MSE  : {mean_squared_error(yr_te, yr_pred):.4f}")
print(f"  RMSE : {np.sqrt(mean_squared_error(yr_te, yr_pred)):.4f}")
print(f"  R²   : {r2_score(yr_te, yr_pred):.4f}")
print("""
  REGRESSION METRIC DEFINITIONS:
  MAE  — mean absolute error; easy to interpret
  MSE  — penalises large errors harder (squares them)
  RMSE — same unit as target; most commonly reported
  R²   — 1.0 = perfect fit; 0 = predicts mean only; <0 = worse than mean
""")


# ============================================================
# CHAPTER 6 — Deep Learning
# ============================================================
print("─" * 60)
print("📘 CHAPTER 6 — Deep Learning (Concept + Lightweight Demo)")
print("─" * 60)

print("""
NEURAL NETWORK ANATOMY:
  Input Layer   → receives raw features
  Hidden Layers → learn abstract representations
  Output Layer  → produces prediction / class probability

KEY CONCEPTS:
  • Activation functions : ReLU, Sigmoid, Softmax
  • Loss functions       : CrossEntropy (clf), MSE (reg)
  • Optimizer            : Adam, SGD — adjusts weights via backprop
  • Epochs               : full passes over training data
  • Batch size           : samples per gradient update
  • Overfitting          : high train acc, low test acc → use Dropout/regularisation
  • Underfitting         : low train acc              → more layers/data

CNN (Convolutional Neural Network):
  Conv Layer → ReLU → Pooling → Flatten → Dense → Output
  Used for images; filters detect edges, shapes, textures.

TESTING DL MODELS — What a Tester checks:
  ✅ Convergence    — loss decreases over epochs
  ✅ Generalisation — val_loss ≈ train_loss (no big gap)
  ✅ Robustness     — performance on noisy/adversarial inputs
  ✅ Latency        — inference time meets SLA
  ✅ Data leakage   — test data never touched during training
""")

# Lightweight neural net simulation (no TF required)
print("🧠 Simulated Training Loop (NumPy — no GPU needed):")
np.random.seed(42)
weights = np.random.randn(4, 3) * 0.1    # 4 features → 3 classes

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y_true):
    n = len(y_true)
    return -np.log(probs[np.arange(n), y_true] + 1e-9).mean()

lr = 0.05
X_nn = X_train_s
y_nn = y_train
loss_history = []

for epoch in range(1, 51):
    logits = X_nn @ weights
    probs  = softmax(logits)
    loss   = cross_entropy(probs, y_nn)
    loss_history.append(loss)
    # Gradient
    delta = probs.copy()
    delta[np.arange(len(y_nn)), y_nn] -= 1
    delta /= len(y_nn)
    weights -= lr * (X_nn.T @ delta)

print(f"  Epoch  1 → Loss: {loss_history[0]:.4f}")
print(f"  Epoch 25 → Loss: {loss_history[24]:.4f}")
print(f"  Epoch 50 → Loss: {loss_history[49]:.4f}")
print("  ✅ Loss is decreasing — model is learning!")


# ============================================================
# CHAPTER 7 — Testing AI Systems
# ============================================================
print("\n" + "─" * 60)
print("📘 CHAPTER 7 — Testing AI Systems")
print("─" * 60)

print("""
UNIQUE CHALLENGES WHEN TESTING AI:
  1. Non-determinism   — stochastic outputs; run multiple times
  2. Oracle problem    — no single 'correct' answer to compare
  3. Data dependency   — model quality = data quality
  4. Explainability    — black-box models hard to validate
  5. Concept drift     — real-world data distribution changes
  6. Bias & fairness   — protected groups may be disadvantaged

AI TESTING STRATEGIES:
  ┌─────────────────────────────────────────────────────────┐
  │ Strategy         │ What it checks                       │
  ├─────────────────────────────────────────────────────────┤
  │ Metamorphic      │ Input transformation → expected      │
  │                  │ output change (e.g. flip image →     │
  │                  │ same class)                          │
  ├─────────────────────────────────────────────────────────┤
  │ Back-to-back     │ Two models on same input; compare    │
  │                  │ outputs (regression testing of AI)   │
  ├─────────────────────────────────────────────────────────┤
  │ A/B Testing      │ Deploy two model versions; compare   │
  │                  │ real-world performance               │
  ├─────────────────────────────────────────────────────────┤
  │ Adversarial      │ Craft inputs to fool the model       │
  │                  │ (security, robustness)               │
  ├─────────────────────────────────────────────────────────┤
  │ Data Validation  │ Schema checks, distribution tests,   │
  │                  │ missing value detection              │
  └─────────────────────────────────────────────────────────┘
""")

# Metamorphic Testing Demo
print("🧪 METAMORPHIC TESTING DEMO")
print("  Rule: Scaling all features by 2x should NOT change class prediction")
X_sample = X_test_s[:5]
pred_original = best_model.predict(X_sample)
pred_scaled   = best_model.predict(X_sample * 2)  # deliberately break rule
violations    = np.sum(pred_original != pred_scaled)
print(f"  Original predictions : {pred_original}")
print(f"  Scaled   predictions : {pred_scaled}")
print(f"  ⚠️  Metamorphic violations found: {violations} (expected non-zero — model is not scale-invariant)")

# Robustness Testing Demo
print("\n🛡️  ROBUSTNESS TESTING DEMO — Adding Gaussian Noise")
results = []
for noise_std in [0, 0.1, 0.5, 1.0, 2.0]:
    noisy = X_test_s + np.random.normal(0, noise_std, X_test_s.shape)
    acc   = accuracy_score(y_test, best_model.predict(noisy))
    results.append((noise_std, acc))
    print(f"  Noise σ={noise_std:.1f} → Accuracy: {acc:.1%}")

print("""
  ✅ A robust model degrades gracefully as noise increases.
  ❌ A brittle model collapses suddenly at small noise levels.
""")

# Data Drift Detection (simple statistical check)
print("📉 DATA DRIFT DETECTION DEMO")
ref_mean    = X_train_s.mean(axis=0)
current_data = X_test_s + 1.5   # simulate drift
curr_mean   = current_data.mean(axis=0)
drift_score  = np.abs(ref_mean - curr_mean)
print(f"  Feature drift (|mean shift|): {np.round(drift_score, 3)}")
drift_features = [iris.feature_names[i] for i, d in enumerate(drift_score) if d > 0.5]
if drift_features:
    print(f"  ⚠️  Significant drift detected in: {drift_features}")
else:
    print("  ✅ No significant drift detected")

# Bias / Fairness Check
print("\n⚖️  FAIRNESS TESTING DEMO")
np.random.seed(42)
n = 200
y_true_fair = np.random.randint(0, 2, n)
group = np.array(["GroupA"] * 100 + ["GroupB"] * 100)
# Introduce bias: GroupB predicted worse
y_pred_fair = y_true_fair.copy()
mistake_idx = np.where(group == "GroupB")[0][:30]
y_pred_fair[mistake_idx] = 1 - y_pred_fair[mistake_idx]

for g in ["GroupA", "GroupB"]:
    mask = group == g
    acc  = accuracy_score(y_true_fair[mask], y_pred_fair[mask])
    rec  = recall_score(y_true_fair[mask], y_pred_fair[mask], zero_division=0)
    print(f"  {g}: Accuracy={acc:.1%}  Recall={rec:.1%}")
print("""
  ✅ Fairness target: metrics should be similar across groups.
  Δ accuracy > 5–10% between groups → potential fairness issue.
""")


# ============================================================
# VISUALISATIONS — All plots in one call
# ============================================================
print("─" * 60)
print("📊 Generating Visualisation Dashboard...")
print("─" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("🎓 ISTQB CT-AI — JK Tech Learning Lab", fontsize=16, fontweight="bold")

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names, ax=axes[0, 0])
axes[0, 0].set_title("Ch5: Confusion Matrix (Random Forest)")
axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("Actual")

# 2. Model Comparison Bar
model_names = list(models.keys())
model_accs  = [accuracy_score(y_test, m.predict(X_test_s)) for m in models.values()]
colors = ["#4e79a7", "#f28e2b", "#59a14f"]
axes[0, 1].bar(model_names, model_accs, color=colors, edgecolor="black")
axes[0, 1].set_ylim(0.8, 1.02); axes[0, 1].set_title("Ch3: Model Accuracy Comparison")
axes[0, 1].set_ylabel("Accuracy"); axes[0, 1].tick_params(axis="x", rotation=15)
for i, v in enumerate(model_accs):
    axes[0, 1].text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=10)

# 3. KMeans Clustering
scatter = axes[0, 2].scatter(X_blob[:, 0], X_blob[:, 1],
                              c=labels, cmap="tab10", alpha=0.7, s=40)
axes[0, 2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   c="red", marker="X", s=200, label="Centroids")
axes[0, 2].set_title("Ch3: KMeans Clustering"); axes[0, 2].legend()

# 4. DL Training Loss
axes[1, 0].plot(range(1, 51), loss_history, color="#e15759", linewidth=2)
axes[1, 0].set_title("Ch6: Neural Net Training Loss")
axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Cross-Entropy Loss")
axes[1, 0].grid(True, alpha=0.3)

# 5. Robustness curve
noise_vals, acc_vals = zip(*results)
axes[1, 1].plot(noise_vals, acc_vals, "o-", color="#76b7b2", linewidth=2, markersize=8)
axes[1, 1].set_title("Ch7: Robustness — Noise vs Accuracy")
axes[1, 1].set_xlabel("Noise σ"); axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_ylim(0, 1.1); axes[1, 1].grid(True, alpha=0.3)

# 6. Fairness Bar
groups     = ["GroupA", "GroupB"]
group_accs = [accuracy_score(y_true_fair[group == g], y_pred_fair[group == g]) for g in groups]
bar_colors = ["#59a14f" if a > 0.85 else "#e15759" for a in group_accs]
axes[1, 2].bar(groups, group_accs, color=bar_colors, edgecolor="black")
axes[1, 2].set_ylim(0, 1.1); axes[1, 2].set_title("Ch7: Fairness — Accuracy by Group")
axes[1, 2].set_ylabel("Accuracy")
axes[1, 2].axhline(y=0.85, color="orange", linestyle="--", label="Min threshold 85%")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/istqb_ai_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Dashboard saved as istqb_ai_dashboard.png")


# ============================================================
# SUMMARY CHEAT SHEET
# ============================================================
print("\n" + "=" * 60)
print("📋  ISTQB CT-AI — QUICK REFERENCE CHEAT SHEET")
print("=" * 60)
print("""
CHAPTER SUMMARY:
  Ch1  Introduction    → AI types, why AI testing differs
  Ch2  AI Techniques   → ML, NLP, Expert Systems, Neural Nets
  Ch3  ML Overview     → Supervised / Unsupervised / RL
  Ch4  ML Data         → Quality, Bias, Preprocessing, Drift
  Ch5  Metrics (Clf)   → Accuracy, Precision, Recall, F1, AUC
  Ch5  Metrics (Reg)   → MAE, MSE, RMSE, R²
  Ch6  Deep Learning   → Layers, CNN, Overfitting, Dropout
  Ch7  Testing AI      → Metamorphic, Robustness, Fairness,
                         Drift, Back-to-back, Adversarial

KEY FORMULAS:
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 × (P × R) / (P + R)
  Accuracy  = (TP + TN) / Total

BIAS TYPES (CRUD):
  C — Confirmation bias
  R — Representation (Sampling) bias
  U — (annotator/label) sUbjectivity bias
  D — Data measurement/collection bias

TESTING STRATEGIES MNEMONIC → "MAB-RAD":
  M — Metamorphic testing
  A — A/B testing
  B — Back-to-back testing
  R — Robustness / adversarial testing
  A — Accuracy & metric threshold testing
  D — Drift & data validation testing
""")
print("=" * 60)
print("🚀  Run complete! All ISTQB CT-AI chapters covered.")
print("    JK Tech Learning Lab — Good luck on the exam! 🎯")
print("=" * 60)
