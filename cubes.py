import os
import csv
import pygame

# generating base dataset with cubes, nothing interesting

OUT_DIR = "raw"
W = H = 128
BG = (30, 30, 40)
CUBE = (220, 90, 60)
N_SIDE = 8          # 8×8 = 64 imgs

os.makedirs(OUT_DIR, exist_ok=True)

pygame.init()
surf = pygame.Surface((W, H))

rows = []
idx = 0
for iy in range(N_SIDE):
    for ix in range(N_SIDE):
        surf.fill(BG)

        # cube pos
        x = int(8 + (W - 16) * ix / (N_SIDE - 1))
        y = int(8 + (H - 16) * iy / (N_SIDE - 1))

        size = 20
        rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
        pygame.draw.rect(surf, CUBE, rect)

        filename = f"cube_{idx:03d}.png"
        path = os.path.join(OUT_DIR, filename)
        pygame.image.save(surf, path)

        # coord = (ix, iy)
        rows.append((filename, float(ix), float(iy)))
        idx += 1

with open("coords.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "x", "y"])
    w.writerows(rows)

pygame.quit()
print(f"Cooked: {idx} images to {OUT_DIR}/, coords.csv written")
