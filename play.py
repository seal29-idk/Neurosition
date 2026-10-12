import os
#os.environ["SDL_VIDEODRIVER"] = "x11"
#os.environ["SDL_RENDER_DRIVER"] = "software"
#os.environ["SDL_FRAMEBUFFER_ACCELERATION"] = "0"
#os.environ["SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR"] = "0"

import math
import time
import torch
import pygame
from model import NeuroRenderer

CKPT = "checkpoints/best.pt" 
WIN = (256, 256)
FPS = 5                # FPS выставлен маленьким чтобы не перегрузить железо (если современное то поставте 24-60 FPS) \ FPS is set low to avoid overloading the hardware (if it's modern, set it to 24-60 FPS)
SPEED = 2.0
RENDER_EVERY = 2 
DEVICE = "cpu"         # У меня нету GPU, использую CPU (поменяйте параметр под свое устройство) \ I don't have a GPU, I use a CPU (change the parameter to match your device)


if not os.path.exists(CKPT):
    print(f"[play.py] ERROR: {CKPT} not found")
    print("[play.py] Run train.py first to create the model(dont forget about dataset)")
    exit(1)

# trained model load
print("[play.py] Model loading...")
ckpt = torch.load(CKPT, map_location=DEVICE)
SIZE = ckpt["size"]
model = NeuroRenderer(out_size=SIZE).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

cmin = ckpt["cmin"].to(DEVICE)
cmax = ckpt["cmax"].to(DEVICE)
print(f"[play.py] model loaded. cmin={cmin.tolist()}, cmax={cmax.tolist()}")

# pygame init 
pygame.init()
screen = pygame.display.set_mode(WIN)
#pygame.display.set_mode(WIN, pygame.SWSURFACE) # если окно не работает попробуйте разкоментировать \ if window dont working try uncomment this
pygame.display.set_caption("Neurosition 2D") # название окна \ window name

# хз как это работает но работает \ idk how its working but working
time.sleep(2)
pygame.event.pump()
pygame.display.flip()
time.sleep(2)
pygame.event.pump()

clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14) # шрифт \ font

pos = ((cmin + cmax) / 2).clone()


def render(x, y):
    xy = torch.tensor([[x, y]], dtype=torch.float32, device=DEVICE)
    xy_n = (xy - cmin) / (cmax - cmin + 1e-6)
    with torch.no_grad():
        out = model(xy_n)[0]
    out = (out.clamp(-1, 1) + 1) * 127.5
    arr = out.permute(1, 2, 0).byte().cpu().numpy()
    surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
    return pygame.transform.scale(surf, WIN)


# some debug (dont used)
#print("[play.py] FirstRenderer...")
#t0 = time.time()
first = render(pos[0].item(), pos[1].item())
#print(f"[play.py] {time.time() - t0:.3f} seconds")

# while(running)
running = True
frame_count = 0
cached_surf = first

try:
    while running:
        dt = clock.tick(FPS) / 1000.0
        frame_count += 1

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        dx = dy = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  dx -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: dx += 1
        if keys[pygame.K_w] or keys[pygame.K_UP]:    dy -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  dy += 1

        if dx or dy:
            n = math.hypot(dx, dy)
            pos[0] += dx / n * SPEED * dt
            pos[1] += dy / n * SPEED * dt

        # R - pos reset
        if keys[pygame.K_r]:
            pos = ((cmin + cmax) / 2).clone()

        if frame_count % RENDER_EVERY == 0:
            try:
                cached_surf = render(pos[0].item(), pos[1].item())
            except Exception as ex:
                print(f"[play.py] renderer error: {ex}")
                # просто продолжает не падая \ just continues

        screen.blit(cached_surf, (0, 0))

        txt = font.render(
            f"x={pos[0]:.2f} y={pos[1]:.2f} WASD|R|ESC",
            True, (255, 255, 0)
        )
        screen.blit(txt, (5, 5)) # текст рендерится \ text render

        #print("b.flip", frame_count, flush=True) old debug
        pygame.display.flip()
        #print("a.flip", frame_count, flush=True) old debug

except KeyboardInterrupt:
    print("\n[play.py] exit by user")

finally:
    print("[play.py] exiting...")
    pygame.quit()
