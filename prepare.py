import os
import subprocess
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# copies from raw to dataset and scales to SIZE*SIZE by nearest 

RAW_DIR = "raw"
OUT_DIR = "dataset"
CSV_IN = "coords.csv"
OUT_PT = "dataset.pt"
SIZE = 128 # Neural image size (SIZE*SIZE)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_IN)
print(f"found {len(df)} raws")

for _, row in tqdm(df.iterrows(), total=len(df), desc="ffmpeg"):
    src = os.path.join(RAW_DIR, row["filename"])
    dst = os.path.join(OUT_DIR, row["filename"])
    if os.path.exists(dst):
        continue
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-vf", f"scale={SIZE}:{SIZE}:flags=neighbor",
        dst
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

frames = []
coords = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="load"):
    path = os.path.join(OUT_DIR, row["filename"])
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    frames.append(arr)
    coords.append([row["x"], row["y"]])

frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
coords = torch.tensor(coords, dtype=torch.float32)

cmin = coords.min(dim=0).values
cmax = coords.max(dim=0).values
coords_n = (coords - cmin) / (cmax - cmin + 1e-6)

torch.save({
    "frames": frames,
    "coords": coords_n,
    "cmin": cmin,
    "cmax": cmax,
}, OUT_PT)

print(f"Saved to {OUT_PT}: {frames.shape}, {coords.shape}")
