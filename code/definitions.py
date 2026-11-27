import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
source_image = BASE_DIR / "data" / "cat.png"

def load_image(path = source_image):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError("Input image not found!")
    if image_bgr.shape != (512, 512, 3):
        raise ValueError("Image must be 512x512 RGB!")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


    rgb = image_rgb.astype(np.float64) / 255.0

    return image_rgb, rgb


def rgb_to_ycbcr(rgb):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) / 1.772 + 0.5
    Cr = (R - Y) / 1.402 + 0.5
    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)
    G = (Y - 0.299 * R - 0.114 * B) / 0.587
    rgb_rec = np.stack(
    [R, G, B],
    axis=2
    )
    rgb_rec = np.clip(rgb_rec, 0.0, 1.0)
    image_rec = np.rint(rgb_rec * 255).astype(np.uint8)
    return image_rec

def calculate_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)   # float64: جلوگیری از overflow
    return (np.inf if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)), mse

def save_gray_channel(channel, dir):

    channel_uint8 = np.rint(
        np.clip(channel, 0.0, 1.0) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        str(dir),
        channel_uint8
    )
    

