from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV  = ROOT / "data/processed/test_external.csv"
MODEL_IN  = ROOT / "models/efficientnet_b0_best.pth"

BATCH_SIZE = 32

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
        return img, y

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

ds = RetinaDataset(TEST_CSV, transform=tfm)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
state = torch.load(MODEL_IN, map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()

y_true, y_pred, p_dr = [], [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        dr_prob = probs[:,1]
        pred = (dr_prob >= 0.5).astype(int)

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
        p_dr.extend(dr_prob.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
p_dr = np.array(p_dr)

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()
total = len(y_true)
acc = (tn+tp)/total
sens = tp/(tp+fn+1e-9)
spec = tn/(tn+fp+1e-9)

print("\nEXTERNAL TEST (MESSIDOR HOLDOUT)")
print("Samples:", total)
print("Confusion Matrix [[TN, FP],[FN, TP]]:\n", cm)
print("\nAccuracy:", round(acc*100,2), "%")
print("Sensitivity (DR):", round(sens*100,2), "%")
print("Specificity (No DR):", round(spec*100,2), "%")
print("\n", classification_report(y_true, y_pred, target_names=["No DR","DR"]))

try:
    print("ROC-AUC:", round(roc_auc_score(y_true, p_dr), 4))
    print("PR-AUC :", round(average_precision_score(y_true, p_dr), 4))
except Exception as e:
    print("AUC error:", e)
