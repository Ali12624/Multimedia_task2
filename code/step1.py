import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "cat.png"
output_dir = BASE_DIR / "results"

image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)

if image_bgr is None:
    raise FileNotFoundError("Input image not found!")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


rgb = image_rgb.astype(np.float64) / 255.0

R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]


# -----------------------------------
# 2. RGB to YCbCr
# -----------------------------------

Y = 0.299 * R + 0.587 * G + 0.114 * B

Cb = (B - Y) / 1.772 + 0.5

Cr = (R - Y) / 1.402 + 0.5


# -----------------------------------
# 3. YCbCr to RGB
# -----------------------------------

R_rec = Y + 1.402 * (Cr - 0.5)

B_rec = Y + 1.772 * (Cb - 0.5)

G_rec = (
    Y - 0.299 * R_rec - 0.114 * B_rec
) / 0.587


# -----------------------------------
# 4. Clip reconstructed RGB
# -----------------------------------

rgb_rec = np.stack(
    [R_rec, G_rec, B_rec],
    axis=2
)

rgb_rec = np.clip(rgb_rec, 0.0, 1.0)


# Convert to uint8.
image_rec = np.rint(rgb_rec * 255).astype(np.uint8)


# -----------------------------------
# 5. Calculate MSE and PSNR
# -----------------------------------

original = image_rgb.astype(np.float64)

reconstructed = image_rec.astype(np.float64)

mse = np.mean((original - reconstructed) ** 2)

if mse == 0:
    psnr = float("inf")
else:
    psnr = 10 * np.log10((255 ** 2) / mse)


print("MSE:", mse)
print("PSNR:", psnr, "dB")


# -----------------------------------
# 6. Save output images
# -----------------------------------

def save_gray_channel(channel, filename):

    channel_uint8 = np.rint(
        np.clip(channel, 0.0, 1.0) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        str(output_dir / filename),
        channel_uint8
    )


save_gray_channel(Y, "stage1/Y.png")
save_gray_channel(Cb, "stage1/Cb.png")
save_gray_channel(Cr, "stage1/Cr.png")


# Save reconstructed RGB image.
cv2.imwrite(
    str(output_dir / "stage1/reconstructed.png"),
    cv2.cvtColor(image_rec, cv2.COLOR_RGB2BGR)
)


# -----------------------------------
# 7. Display results
# -----------------------------------

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Original RGB")

axes[0, 1].imshow(Y, cmap="gray", vmin=0, vmax=1)
axes[0, 1].set_title("Y - Luminance")

axes[0, 2].imshow(Cb, cmap="gray", vmin=0, vmax=1)
axes[0, 2].set_title("Cb - Chrominance")

axes[1, 0].imshow(Cr, cmap="gray", vmin=0, vmax=1)
axes[1, 0].set_title("Cr - Chrominance")

axes[1, 1].imshow(image_rec)
axes[1, 1].set_title("Reconstructed RGB")

axes[1, 2].imshow(
    np.abs(image_rgb.astype(np.int16) -
           image_rec.astype(np.int16)),
    cmap="gray"
)
axes[1, 2].set_title("Absolute Difference")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    output_dir / "stage1/stage1_results.png",
    dpi=150
)

plt.show()

