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
    

def zigzag_indices(size=8):

    indices = []

    for diagonal in range(2 * size - 1):

        if diagonal % 2 == 0:

            row = min(diagonal, size - 1)

            col = diagonal - row

            while row >= 0 and col < size:

                indices.append((row, col))

                row -= 1
                col += 1

        else:

            col = min(diagonal, size - 1)

            row = diagonal - col

            while col >= 0 and row < size:

                indices.append((row, col))

                row += 1
                col -= 1

    return indices


def apply_dct(channel):

    height, width = channel.shape

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("Channel dimensions must be divisible by 8!")

    # Convert [0, 1] to [0, 255]
    channel_255 = channel * 255.0

    # JPEG Level Shift
    shifted = channel_255 - 128.0

    # Keep all values as float64
    shifted = shifted.astype(np.float64)

    # Prepare output array
    dct_result = np.zeros_like(shifted)

    # Apply DCT to each 8x8 block
    for row in range(0, height, 8):

        for col in range(0, width, 8):

            block = shifted[row:row + 8, col:col + 8]

            dct_block = cv2.dct(block)

            dct_result[row:row + 8, col:col + 8] = dct_block

    return dct_result


def keep_n_coefficients(dct_block, N, zigzag):

    if N < 1 or N > 64:
        raise ValueError("N must be between 1 and 64!")

    # Create a zero matrix
    filtered = np.zeros_like(dct_block)

    # Keep first N coefficients in Zigzag order
    for row, col in zigzag[:N]:

        filtered[row, col] = dct_block[row, col]

    return filtered

def apply_idct(dct_coefficients):

    height, width = dct_coefficients.shape

    reconstructed = np.zeros_like(dct_coefficients)

    for row in range(0, height, 8):

        for col in range(0, width, 8):

            block = dct_coefficients[
                row:row + 8,
                col:col + 8
            ]

            idct_block = cv2.idct(block)

            reconstructed[
                row:row + 8,
                col:col + 8
            ] = idct_block

    # Reverse JPEG Level Shift
    reconstructed = (reconstructed + 128.0) / 255.0

    return reconstructed

def process_channel(channel, N, zigzag):

    # Step 1: DCT
    dct_coefficients = apply_dct(channel)

    height, width = dct_coefficients.shape

    # Step 2: Keep N coefficients in each 8x8 block
    filtered = np.zeros_like(dct_coefficients)

    for row in range(0, height, 8):

        for col in range(0, width, 8):

            block = dct_coefficients[
                row:row + 8,
                col:col + 8
            ]

            filtered_block = keep_n_coefficients(
                block,
                N,
                zigzag
            )

            filtered[row:row + 8, col:col + 8] = filtered_block

    # Step 3: IDCT
    reconstructed = apply_idct(filtered)

    return reconstructed

