from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VAL_CSV   = ROOT / "data/processed/val.csv"
MODEL_IN  = ROOT / "models/efficientnet_b0_best.pth"

BATCH_SIZE = 32

# Device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Device:", device)

class RetinaDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        y = int(row["label"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y, img_path

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

ds = RetinaDataset(VAL_CSV, transform=val_tfms)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
state = torch.load(MODEL_IN, map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()

y_true, y_pred, conf, p_dr, paths = [], [], [], [], []

with torch.no_grad():
    for x, y, p in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        # prob of DR = class 1
        dr_prob = probs[:, 1]
        pred = (dr_prob >= 0.5).astype(int)
        mx_conf = probs.max(axis=1)

        y_true.extend(list(y))
        y_pred.extend(list(pred))
        p_dr.extend(list(dr_prob))
        conf.extend(list(mx_conf))
        paths.extend(list(p))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
p_dr   = np.array(p_dr)
conf   = np.array(conf)

# --- Confusion matrix ---
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()
total = len(y_true)
acc = (tn + tp) / total
sensitivity = tp / (tp + fn + 1e-9)   # recall for DR
specificity = tn / (tn + fp + 1e-9)   # recall for No DR
precision_dr = tp / (tp + fp + 1e-9)
f1_dr = 2 * precision_dr * sensitivity / (precision_dr + sensitivity + 1e-9)

print("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n", cm)
print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=["No DR", "DR"]))

summary = pd.DataFrame([
    ("Samples", total, "100%"),
    ("Accuracy", (tn+tp), f"{acc*100:.2f}%"),
    ("Sensitivity/Recall (DR)", tp, f"{sensitivity*100:.2f}%"),
    ("Specificity (No DR)", tn, f"{specificity*100:.2f}%"),
    ("Precision (DR)", tp, f"{precision_dr*100:.2f}%"),
    ("F1 (DR)", "-", f"{f1_dr*100:.2f}%"),
    ("TN", tn, f"{tn/total*100:.2f}%"),
    ("FP", fp, f"{fp/total*100:.2f}%"),
    ("FN", fn, f"{fn/total*100:.2f}%"),
    ("TP", tp, f"{tp/total*100:.2f}%"),
], columns=["Metric", "Count", "Percent/Value"])

print("\n", summary.to_string(index=False))

# --- ROC AUC / PR AUC ---
try:
    roc_auc = roc_auc_score(y_true, p_dr)
    pr_auc = average_precision_score(y_true, p_dr)
except Exception as e:
    roc_auc, pr_auc = None, None
    print("AUC computation issue:", e)

print("\nROC-AUC:", None if roc_auc is None else round(roc_auc, 4))
print("PR-AUC :", None if pr_auc is None else round(pr_auc, 4))

# --- Plots ---
# 1) Confusion matrix (raw + normalized)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Counts)")
plt.xticks([0,1], ["No DR", "DR"])
plt.yticks([0,1], ["No DR", "DR"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(cm_norm, interpolation="nearest")
plt.title("Confusion Matrix (Normalized)")
plt.xticks([0,1], ["No DR", "DR"])
plt.yticks([0,1], ["No DR", "DR"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm_norm[i,j]*100:.1f}%", ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# 2) ROC curve
fpr, tpr, _ = roc_curve(y_true, p_dr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.show()

# 3) Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_true, p_dr)
plt.figure(figsize=(6,5))
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()

# 4) Confidence histograms (correct vs wrong)
is_correct = (y_true == y_pred)
plt.figure(figsize=(7,5))
plt.hist(conf[is_correct], bins=20, alpha=0.7, label="Correct")
plt.hist(conf[~is_correct], bins=20, alpha=0.7, label="Wrong")
plt.title("Confidence Distribution")
plt.xlabel("Max softmax confidence")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

# 5) Calibration / reliability plot (10 bins)
bins = np.linspace(0, 1, 11)
bin_ids = np.digitize(p_dr, bins) - 1
bin_acc = []
bin_conf = []
for b in range(10):
    mask = bin_ids == b
    if mask.sum() == 0:
        continue
    bin_acc.append((y_true[mask] == (p_dr[mask] >= 0.5)).mean())
    bin_conf.append(p_dr[mask].mean())

plt.figure(figsize=(6,5))
plt.plot([0,1],[0,1])
plt.scatter(bin_conf, bin_acc)
plt.title("Calibration (Reliability Plot)")
plt.xlabel("Mean predicted prob (DR)")
plt.ylabel("Empirical accuracy")
plt.tight_layout()
plt.show()

# 6) Error analysis grids: most confident wrong + least confident correct
def show_grid(indices, title):
    # denorm
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    plt.figure(figsize=(14, 7))
    for i, idx in enumerate(indices, 1):
        img = Image.open(paths[idx]).convert("RGB")
        x = val_tfms(img)
        x_vis = (x * std + mean).clamp(0,1)

        plt.subplot(2, 3, i)
        plt.imshow(x_vis.permute(1,2,0))
        plt.axis("off")
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]} pDR:{p_dr[idx]:.2f} conf:{conf[idx]:.2f}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

wrong_idx = np.where(~is_correct)[0]
correct_idx = np.where(is_correct)[0]

if len(wrong_idx) >= 6:
    most_conf_wrong = wrong_idx[np.argsort(conf[wrong_idx])[::-1][:6]]
    show_grid(most_conf_wrong, "Most Confident Wrong Predictions (Review these)")

if len(correct_idx) >= 6:
    least_conf_correct = correct_idx[np.argsort(conf[correct_idx])[:6]]
    show_grid(least_conf_correct, "Least Confident Correct Predictions (Borderline cases)")
