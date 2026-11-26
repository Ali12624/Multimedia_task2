import cv2
import numpy as np
import matplotlib.pyplot as plt
from definitions import *
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

output_dir = BASE_DIR / "results" / "stage2"

output_dir.mkdir(parents=True, exist_ok=True) # ==> make directory if not exists


# -----------------------------------
# 3. Chroma Subsampling 4:2:0
# -----------------------------------

def chroma_subsampling(channel):

    height, width = channel.shape

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Image dimensions must be even!")

    # Divide image into 2x2 blocks.
    blocks = channel.reshape(
        height // 2,
        2,
        width // 2,
        2
    )

    # Calculate the average of each 2x2 block.
    subsampled = blocks.mean(axis=(1, 3))

    return subsampled

def nearest_neighbor_upsampling(channel):

    upsampled = np.repeat(
        np.repeat(channel, 2, axis=0),
        2,
        axis=1
    )

    return upsampled



def main():
    image_rgb, rgb = load_image()

    # RGB to YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(rgb)
   
   
#    --------
#    Subsampling
#    --------
    Cb_sub = chroma_subsampling(Cb)

    Cr_sub = chroma_subsampling(Cr)

#    ---------
#    Nearest Neighbor Unsampling
#    ---------
    Cb_up = nearest_neighbor_upsampling(Cb_sub)

    Cr_up = nearest_neighbor_upsampling(Cr_sub)
    
    # --------
    # Reconstruct RGB
    # --------
    image_rec = ycbcr_to_rgb(Y, Cb_up, Cr_up)
    
    # ---------
    # Calculate MSE and PSNR
    # ---------
    original = image_rgb.astype(np.float64)

    reconstructed = image_rec.astype(np.float64)
    
    psnr, mse = calculate_psnr(original, reconstructed)
    
    print(f"MSE: {mse}")
    print(f"PSNR: {psnr}")
    
    
    save_gray_channel(Y, f"{str(output_dir)}/Y.png")

    save_gray_channel(Cb, f"{str(output_dir)}/Cb_original.png")
    save_gray_channel(Cr, f"{str(output_dir)}/Cr_original.png")

    save_gray_channel(Cb_sub, f"{str(output_dir)}/Cb_subsampled.png")
    save_gray_channel(Cr_sub, f"{str(output_dir)}/Cr_subsampled.png")

    save_gray_channel(Cb_up, f"{str(output_dir)}/Cb_upsampled.png")
    save_gray_channel(Cr_up, f"{str(output_dir)}/Cr_upsampled.png")
    cv2.imwrite(
        str(output_dir / "reconstructed.png"),
        cv2.cvtColor(image_rec, cv2.COLOR_RGB2BGR)
)
    # -------
    # Display results
    # -------

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original RGB")

    axes[0, 1].imshow(Y, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Y")

    axes[0, 2].imshow(Cb_sub, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Cb Subsampled")

    axes[0, 3].imshow(Cr_sub, cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title("Cr Subsampled")

    axes[1, 0].imshow(Cb_up, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Cb Upsampled")

    axes[1, 1].imshow(Cr_up, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Cr Upsampled")

    axes[1, 2].imshow(image_rec)
    axes[1, 2].set_title("Reconstructed RGB")

    difference = np.abs(image_rgb.astype(np.int16) - image_rec.astype(np.int16))
    print("max abs diff:", difference.max())
    print("pixels with any difference:", np.count_nonzero(difference.any(axis=2)))

    amplified = np.clip(difference * 10, 0, 255).astype(np.uint8)   # تقویت ×10 برای دیده شدن
    axes[1, 3].imshow(amplified)
    axes[1, 3].set_title("Absolute Difference (x10)")


    for ax in axes.flat:
        ax.axis("off")


    plt.tight_layout()

    plt.savefig(
        output_dir / "stage2_results.png",
        dpi=150
    )

    plt.show()
    
    
    # ---------
    # Calculate theoretical data size
    # ---------

    height, width = image_rgb.shape[:2]

    rgb_size = height * width * 3

    ycbcr_444_size = height * width * 3

    ycbcr_420_size = (
        height * width
        + (height // 2) * (width // 2) * 2
    )
    
    def print_size(name, size):

        print(f"\n{name}")
        print("Size in bytes:", size)
        print("Size in KB:", size / 1024)

    print_size("RGB 24-bit", rgb_size)

    print_size(f"YCbCr 4:4:4", ycbcr_444_size)

    print_size(f"YCbCr 4:2:0", ycbcr_420_size)


    reduction = (
        (rgb_size - ycbcr_420_size) / rgb_size
    ) * 100

    print("\nSize reduction:", reduction, "%")

if __name__ == "__main__":
    # print(rgb)
    main()
    
    
    