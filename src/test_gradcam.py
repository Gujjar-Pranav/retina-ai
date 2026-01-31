from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

from gradcam import GradCAM
from find_last_conv import find_last_conv_layer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/efficientnet_b0_best.pth"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Device:", device)

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# Pick any cached image
sample_img = next((ROOT / "data/cache_224/aptos").glob("*.jpg"))
img = Image.open(sample_img).convert("RGB")

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()

x = tfm(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
pred = int(np.argmax(probs))
conf = float(np.max(probs))
print("Pred:", pred, "Conf:", conf, "pDR:", float(probs[1]))

target_layer = find_last_conv_layer(model)
cam = GradCAM(model, target_layer)(x, class_idx=pred)

## Make overlay (force same shape + 3 channels)
img_resized = np.array(img.resize((224,224))).astype(np.uint8)          # H,W,3 RGB
heat = (cam * 255).astype(np.uint8)                                     # H,W
heat = cv2.resize(heat, (224,224))                                      # ensure 224x224
heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)                  # H,W,3 BGR

# Convert RGB->BGR for OpenCV blending, then back for plotting
img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)                  # H,W,3 BGR
overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heat_color, 0.4, 0)         # blend
overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)                  # back to RGB

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img_resized); plt.axis("off"); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(cam, cmap="jet"); plt.axis("off"); plt.title("Grad-CAM")
plt.subplot(1,3,3); plt.imshow(overlay); plt.axis("off"); plt.title("Overlay")
plt.tight_layout()
plt.show()
