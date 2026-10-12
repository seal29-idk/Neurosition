import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import NeuroRenderer

DATA = "dataset.pt"
CKPT_DIR = "checkpoints"
EPOCHS = 256
BATCH = 32
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CKPT_DIR, exist_ok=True)

data = torch.load(DATA)
frames = data["frames"]
coords = data["coords"]
print(f"Данных: {frames.shape[0]}, устройство: {DEVICE}")

ds = TensorDataset(coords, frames)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

model = NeuroRenderer(out_size=frames.shape[-1]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.MSELoss()

best = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    pbar = tqdm(dl, desc=f"epoch {epoch}/{EPOCHS}")
    for c, f in pbar:
        c, f = c.to(DEVICE), f.to(DEVICE)
        pred = model(c)
        loss = loss_fn(pred, f)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * c.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    sched.step()
    avg = total / len(ds)
    print(f"epoch {epoch}: MSE={avg:.5f}")

    if avg < best:
        best = avg
        torch.save({
            "model": model.state_dict(),
            "cmin": data["cmin"],
            "cmax": data["cmax"],
            "size": frames.shape[-1],
        }, os.path.join(CKPT_DIR, "best.pt"))

print(f"Готово. Лучший MSE: {best:.5f}")
