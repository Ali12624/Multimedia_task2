import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from step2 import chroma_subsampling, nearest_neighbor_upsampling
from definitions import (
    load_image,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    calculate_psnr,
    save_gray_channel
)

# --------
# Loading Image
# --------
BASE_DIR = Path(__file__).resolve().parent.parent

output_dir = BASE_DIR / "results" / "stage3"

output_dir.mkdir(parents=True, exist_ok=True)

image_rgb, rgb = load_image()


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

def display_results(image_rgb, results):

    N_values = [1, 2, 4, 8, 16, 32, 64]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 9)
    )

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image")

    for i, N in enumerate(N_values):

        row = (i + 1) // 4
        col = (i + 1) % 4

        image = results[N]["image"]

        psnr = results[N]["psnr"]

        axes[row, col].imshow(image)

        axes[row, col].set_title(
            f"N = {N}\nPSNR = {psnr:.2f} dB"
        )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_dir / "comparison.png",
        dpi=150
    )

    plt.show()

def main():




    # -----------------------------------
    # 2. RGB to YCbCr
    # -----------------------------------

    Y, Cb, Cr = rgb_to_ycbcr(rgb)


    # -----------------------------------
    # 3. Chroma Subsampling 4:2:0
    # -----------------------------------

    Cb_sub = chroma_subsampling(Cb)

    Cr_sub = chroma_subsampling(Cr)


    # -----------------------------------
    # 4. Zigzag
    # -----------------------------------

    zigzag = zigzag_indices()


    # -----------------------------------
    # 5. Different N values
    # -----------------------------------

    N_values = [1, 2, 4, 8, 16, 32, 64]

    results = {}


    for N in N_values:

        print(f"\nProcessing N = {N}")

        # DCT + Coefficient Selection + IDCT

        Y_rec = process_channel(Y, N, zigzag)

        Cb_rec = process_channel(Cb_sub, N, zigzag)

        Cr_rec = process_channel(Cr_sub, N, zigzag)


        # -----------------------------------
        # 6. Chroma Upsampling
        # -----------------------------------

        Cb_up = nearest_neighbor_upsampling(Cb_rec)

        Cr_up = nearest_neighbor_upsampling(Cr_rec)


        # -----------------------------------
        # 7. Reconstruct RGB
        # -----------------------------------

        image_rec = ycbcr_to_rgb(
            Y_rec,
            Cb_up,
            Cr_up
        )


        # -----------------------------------
        # 8. Calculate PSNR and MSE
        # -----------------------------------

        psnr, mse = calculate_psnr(
            image_rgb,
            image_rec
        )


        print(f"N = {N}")
        print(f"MSE  = {mse:.6f}")
        print(f"PSNR = {psnr:.4f} dB")


        # -----------------------------------
        # 9. Save reconstructed image
        # -----------------------------------

        filename = f"reconstructed_N{N}.png"

        cv2.imwrite(
            str(output_dir / filename),
            cv2.cvtColor(image_rec, cv2.COLOR_RGB2BGR)
        )


        results[N] = {
            "image": image_rec,
            "psnr": psnr,
            "mse": mse
        }
        
    display_results(image_rgb, results)

        
    # -----------------------------------
    # 10. Theoretical Data Size
    # -----------------------------------

    total_blocks = 4096 + 1024 + 1024

    print("\nTheoretical Data Size")

    for N in N_values:

        total_coefficients = total_blocks * N

        size_bytes = total_coefficients * 4

        size_kb = size_bytes / 1024

        print(
            f"N={N:2d} | "
            f"Coefficients={total_coefficients:7d} | "
            f"Size={size_bytes:9d} Bytes | "
            f"{size_kb:7.2f} KB"
        )
        
        
if __name__ == "__main__":
    main()