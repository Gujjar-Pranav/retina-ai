from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import random

ROOT = Path(__file__).resolve().parents[1]
VAL_CSV   = ROOT / "data/processed/val.csv"
MODEL_IN  = ROOT / "models/efficientnet_b0_best.pth"

IMG_SIZE = 224
BATCH_SIZE = 32

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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

y_true, y_pred, y_conf, paths = [], [], [], []

with torch.no_grad():
    for x, y, p in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        y_true.extend(y)
        y_pred.extend(pred.cpu().tolist())
        y_conf.extend(conf.cpu().tolist())
        paths.extend(p)

# Metrics
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()
total = len(y_true)
acc = (tn + tp) / total

print("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n", cm)
print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=["No DR", "DR"]))

# Table with counts + %
rows = [
    ("Val samples", total, "100%"),
    ("Accuracy", (tn+tp), f"{acc*100:.2f}%"),
    ("True Negative (No DR correctly)", tn, f"{tn/total*100:.2f}%"),
    ("False Positive (No DR → DR)", fp, f"{fp/total*100:.2f}%"),
    ("False Negative (DR → No DR)", fn, f"{fn/total*100:.2f}%"),
    ("True Positive (DR correctly)", tp, f"{tp/total*100:.2f}%"),
]
df = pd.DataFrame(rows, columns=["Metric", "Count", "Percent"])
print("\n", df.to_string(index=False))

# ---- 6-image grid preview ----
# Pick 6 random samples
idxs = random.sample(range(total), 6)

# helper: denormalize for display
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

plt.figure(figsize=(14, 8))
for i, idx in enumerate(idxs, 1):
    img = Image.open(paths[idx]).convert("RGB")
    x = val_tfms(img)
    x_vis = (x * std + mean).clamp(0,1)

    true = y_true[idx]
    pred = y_pred[idx]
    conf = y_conf[idx]

    plt.subplot(2, 3, i)
    plt.imshow(x_vis.permute(1,2,0))
    plt.axis("off")
    plt.title(f"T:{true}  P:{pred}  C:{conf*100:.1f}%")

plt.tight_layout()
plt.show()
