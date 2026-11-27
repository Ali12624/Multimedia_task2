
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from definitions import (
    load_image,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    calculate_psnr
)

from step2 import (
    chroma_subsampling,
    nearest_neighbor_upsampling
)

from step3 import (
    apply_dct,
    apply_idct
)

# ==========================================
# 1. Directories
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

output_dir = BASE_DIR / "results" / "stage4"

output_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. JPEG Quantization Tables
# ==========================================

# Luminance Quantization Table

Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)


# Chrominance Quantization Table

Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float64)




# ==========================================
# 4. Quantization
# ==========================================

def quantize_channel(dct_coefficients, Q):

    height, width = dct_coefficients.shape

    quantized = np.zeros_like(dct_coefficients)

    for row in range(0, height, 8):

        for col in range(0, width, 8):

            block = dct_coefficients[
                row:row + 8,
                col:col + 8
            ]

            # Quantization
            quantized_block = np.rint(block / Q)

            quantized[
                row:row + 8,
                col:col + 8
            ] = quantized_block

    return quantized


# ==========================================
# 5. Dequantization
# ==========================================

def dequantize_channel(quantized, Q):

    height, width = quantized.shape

    dequantized = np.zeros_like(quantized)

    for row in range(0, height, 8):

        for col in range(0, width, 8):

            block = quantized[
                row:row + 8,
                col:col + 8
            ]

            # Dequantization
            dequantized_block = block * Q

            dequantized[
                row:row + 8,
                col:col + 8
            ] = dequantized_block

    return dequantized


# ==========================================
# 6. Process One Channel
# ==========================================

def process_channel_quantization(channel, Q):

    # Step 1: DCT
    dct_coefficients = apply_dct(channel)

    # Step 2: Quantization
    quantized = quantize_channel(
        dct_coefficients,
        Q
    )

    # Step 3: Dequantization
    dequantized = dequantize_channel(
        quantized,
        Q
    )

    # Step 4: IDCT
    reconstructed = apply_idct(dequantized)

    return reconstructed, dct_coefficients, quantized


# ==========================================
# 7. Analyze Zero Coefficients
# ==========================================

def analyze_coefficients(dct, quantized):

    total = dct.size

    # All zero coefficients after quantization
    zero_after = np.count_nonzero(
        quantized == 0
    )

    # Originally non-zero coefficients
    originally_nonzero = np.count_nonzero(
        dct != 0
    )

    # Coefficients changed from non-zero to zero
    newly_zeroed = np.count_nonzero(
        (dct != 0) & (quantized == 0)
    )

    # Percentage of all zero coefficients
    zero_percentage = (
        zero_after / total
    ) * 100

    # Percentage of newly zeroed coefficients
    newly_zeroed_percentage = (
        newly_zeroed / total
    ) * 100

    return {
        "total": total,
        "zero_after": zero_after,
        "newly_zeroed": newly_zeroed,
        "zero_percentage": zero_percentage,
        "newly_zeroed_percentage": newly_zeroed_percentage
    }


# ==========================================
# 8. Calculate Theoretical Data Size
# ==========================================

def calculate_theoretical_size(*quantized_channels):

    total = sum(
        channel.size
        for channel in quantized_channels
    )

    nonzero = sum(
        np.count_nonzero(channel)
        for channel in quantized_channels
    )

    zero = total - nonzero

    # Each quantized coefficient = 16 bits = 2 bytes
    bytes_per_coefficient = 2

    # All coefficients are counted, including zeros
    estimated_size = (
        total * bytes_per_coefficient
    )

    return {
        "total_coefficients": total,
        "nonzero_coefficients": nonzero,
        "zero_coefficients": zero,
        "estimated_size": estimated_size,
        "size_kb": estimated_size / 1024
    }


# ==========================================
# 9. Save Grayscale Channel
# ==========================================

def save_channel(channel, filename):

    channel_uint8 = np.rint(
        np.clip(channel, 0.0, 1.0) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        str(output_dir / filename),
        channel_uint8
    )


# ==========================================
# 10. Display Results
# ==========================================

def display_results(
    image_rgb,
    Y,
    Cb_sub,
    Cr_sub,
    Y_rec,
    Cb_up,
    Cr_up,
    image_rec,
    psnr
):

    # --------------------------------------
    # Figure 1: Channel Results
    # --------------------------------------

    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 9)
    )

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original RGB")

    axes[0, 1].imshow(
        Y,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 1].set_title("Original Y")

    axes[0, 2].imshow(
        Cb_sub,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 2].set_title("Cb Subsampled")

    axes[0, 3].imshow(
        Cr_sub,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 3].set_title("Cr Subsampled")

    axes[1, 0].imshow(
        Y_rec,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1, 0].set_title("Reconstructed Y")

    axes[1, 1].imshow(
        Cb_up,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1, 1].set_title("Reconstructed Cb")

    axes[1, 2].imshow(
        Cr_up,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1, 2].set_title("Reconstructed Cr")

    axes[1, 3].imshow(image_rec)
    axes[1, 3].set_title("Reconstructed RGB")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_dir / "stage4_channels.png",
        dpi=150
    )

    plt.show()


    # --------------------------------------
    # Figure 2: Original vs Reconstructed
    # --------------------------------------

    difference = np.abs(
        image_rgb.astype(np.int16) -
        image_rec.astype(np.int16)
    )

    amplified = np.clip(
        difference * 5,
        0,
        255
    ).astype(np.uint8)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original RGB")

    axes[1].imshow(image_rec)
    axes[1].set_title(
        f"JPEG Quantization\nPSNR = {psnr:.2f} dB"
    )

    axes[2].imshow(amplified)
    axes[2].set_title("Absolute Difference (x5)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_dir / "stage4_comparison.png",
        dpi=150
    )

    plt.show()
    

# ==========================================
# 12. Stage 5: Data Size Comparison
# ==========================================

def calculate_stage5_size(
    original_size,
    Y_dct,
    Cb_dct,
    Cr_dct,
    stage4_size
):

    # --------------------------------------
    # Calculate Number of DCT Blocks
    # --------------------------------------

    blocks_Y = (Y_dct.shape[0] // 8) * (Y_dct.shape[1] // 8)

    blocks_Cb = (Cb_dct.shape[0] // 8) * (Cb_dct.shape[1] // 8)

    blocks_Cr = (Cr_dct.shape[0] // 8) * (Cr_dct.shape[1] // 8)

    total_blocks = blocks_Y + blocks_Cb + blocks_Cr

    print("\n================================")
    print("STAGE 5: DATA SIZE COMPARISON")
    print("================================")

    print("\nDCT Block Analysis")

    print("Y blocks:", blocks_Y)
    print("Cb blocks:", blocks_Cb)
    print("Cr blocks:", blocks_Cr)
    print("Total blocks:", total_blocks)

    print("\nOriginal Image Size")
    print("Original size:", original_size, "Bytes")
    print(f"Original size: {original_size / 1024:.2f} KB")

    # --------------------------------------
    # Stage 3: Different N Values
    # --------------------------------------

    N_values = [1, 2, 4, 8, 16, 32, 64]

    stage3_results = []

    for N in N_values:

        total_coefficients = total_blocks * N

        # Each DCT coefficient = 4 bytes
        size_bytes = total_coefficients * 4

        size_kb = size_bytes / 1024

        # Percentage of size change
        size_change = (
            (size_bytes - original_size) / original_size
        ) * 100

        # Compression ratio
        compression_ratio = original_size / size_bytes

        stage3_results.append({
            "N": N,
            "coefficients": total_coefficients,
            "size_bytes": size_bytes,
            "size_kb": size_kb,
            "size_change": size_change,
            "compression_ratio": compression_ratio
        })

    # --------------------------------------
    # Stage 4: JPEG Quantization
    # --------------------------------------

    stage4_coefficients = total_blocks * 64

    stage4_size_kb = stage4_size / 1024

    stage4_change = (
        (stage4_size - original_size) / original_size
    ) * 100

    stage4_ratio = original_size / stage4_size

    # --------------------------------------
    # Print Stage 3 Results
    # --------------------------------------

    print("\n================================")
    print("STAGE 3: DCT COEFFICIENT SELECTION")
    print("================================")

    print(
        f"{'N':<8}"
        f"{'Coefficients':<18}"
        f"{'Size (Bytes)':<18}"
        f"{'Size (KB)':<14}"
        f"{'Change (%)':<14}"
        f"{'Ratio':<10}"
    )

    print("-" * 82)

    for result in stage3_results:

        print(
            f"{result['N']:<8}"
            f"{result['coefficients']:<18}"
            f"{result['size_bytes']:<18}"
            f"{result['size_kb']:<14.2f}"
            f"{result['size_change']:<14.2f}"
            f"{result['compression_ratio']:.2f}:1"
        )

    # --------------------------------------
    # Print Stage 4 Results
    # --------------------------------------

    print("\n================================")
    print("STAGE 4: JPEG QUANTIZATION")
    print("================================")

    print("Total coefficients:", stage4_coefficients)

    print("Size (Bytes):", stage4_size)

    print(f"Size (KB): {stage4_size_kb:.2f}")

    print(f"Size change: {stage4_change:.2f}%")

    print(f"Compression ratio: {stage4_ratio:.2f}:1")

    # --------------------------------------
    # Save Results
    # --------------------------------------

    return {
        "original_size": original_size,
        "total_blocks": total_blocks,
        "stage3_results": stage3_results,
        "stage4": {
            "coefficients": stage4_coefficients,
            "size_bytes": stage4_size,
            "size_kb": stage4_size_kb,
            "size_change": stage4_change,
            "compression_ratio": stage4_ratio
        }
    }


# ==========================================
# 13. Plot Stage 5 Results
# ==========================================

def plot_stage5_results(results):

    stage3_results = results["stage3_results"]

    N_values = [
        item["N"]
        for item in stage3_results
    ]

    stage3_sizes = [
        item["size_kb"]
        for item in stage3_results
    ]

    stage4_size = results["stage4"]["size_kb"]

    original_size = results["original_size"] / 1024

    # --------------------------------------
    # Figure 1: Stage 3 Size Comparison
    # --------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        N_values,
        stage3_sizes,
        marker="o",
        linewidth=2,
        label="Stage 3"
    )

    plt.axhline(
        y=original_size,
        linestyle="--",
        label="Original Image"
    )

    plt.axhline(
        y=stage4_size,
        linestyle=":",
        label="Stage 4"
    )

    plt.xlabel("Number of DCT Coefficients (N)")

    plt.ylabel("Data Size (KB)")

    plt.title("Data Size Comparison")

    plt.xticks(N_values)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "stage5_size_comparison.png",
        dpi=150
    )

    plt.show()

    # --------------------------------------
    # Figure 2: Stage 3 and Stage 4
    # --------------------------------------

    labels = [
        f"N = {item['N']}"
        for item in stage3_results
    ]

    sizes = stage3_sizes.copy()

    labels.append("Stage 4")
    sizes.append(stage4_size)

    plt.figure(figsize=(11, 6))

    plt.bar(labels, sizes)

    plt.axhline(
        y=original_size,
        linestyle="--",
        label="Original Image"
    )

    plt.xlabel("Compression Method")

    plt.ylabel("Data Size (KB)")

    plt.title("Original, Stage 3 and Stage 4 Size Comparison")

    plt.grid(axis="y")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "stage5_size_bar_chart.png",
        dpi=150
    )

    plt.show()

    


# ==========================================
# 11. Main
# ==========================================

def main():

    # --------------------------------------
    # Load Original Image
    # --------------------------------------

    image_rgb, rgb = load_image()
    original_size = image_rgb.nbytes
    print("Image Shape:", image_rgb.shape)


    # --------------------------------------
    # RGB to YCbCr
    # --------------------------------------

    Y, Cb, Cr = rgb_to_ycbcr(rgb)


    # --------------------------------------
    # Chroma Subsampling 4:2:0
    # --------------------------------------

    Cb_sub = chroma_subsampling(Cb)

    Cr_sub = chroma_subsampling(Cr)


    # --------------------------------------
    # DCT + Quantization + IDCT
    # --------------------------------------

    Y_rec, Y_dct, Y_quant = process_channel_quantization(
        Y,
        Q_Y
    )

    Cb_rec, Cb_dct, Cb_quant = process_channel_quantization(
        Cb_sub,
        Q_C
    )

    Cr_rec, Cr_dct, Cr_quant = process_channel_quantization(
        Cr_sub,
        Q_C
    )


    # --------------------------------------
    # Stage 3 Comparison: N = 64
    # Without Quantization
    # --------------------------------------

    Y_stage3 = apply_idct(Y_dct)

    Cb_stage3 = apply_idct(Cb_dct)

    Cr_stage3 = apply_idct(Cr_dct)

    Cb_stage3_up = nearest_neighbor_upsampling(
        Cb_stage3
    )

    Cr_stage3_up = nearest_neighbor_upsampling(
        Cr_stage3
    )

    image_stage3 = ycbcr_to_rgb(
        Y_stage3,
        Cb_stage3_up,
        Cr_stage3_up
    )

    psnr_stage3, mse_stage3 = calculate_psnr(
        image_rgb,
        image_stage3
    )


    # --------------------------------------
    # Chroma Upsampling
    # --------------------------------------

    Cb_up = nearest_neighbor_upsampling(Cb_rec)

    Cr_up = nearest_neighbor_upsampling(Cr_rec)


    # --------------------------------------
    # YCbCr to RGB
    # --------------------------------------

    image_rec = ycbcr_to_rgb(
        Y_rec,
        Cb_up,
        Cr_up
    )


    # --------------------------------------
    # Calculate MSE and PSNR
    # --------------------------------------

    psnr, mse = calculate_psnr(
        image_rgb,
        image_rec
    )

    print("\n================================")
    print("STAGE 4: JPEG QUANTIZATION")
    print("================================")

    print(f"MSE  = {mse:.6f}")
    print(f"PSNR = {psnr:.4f} dB")


    # --------------------------------------
    # Save Three Reconstructed Channels
    # --------------------------------------

    save_channel(
        Y_rec,
        "Y_reconstructed.png"
    )

    save_channel(
        Cb_up,
        "Cb_reconstructed.png"
    )

    save_channel(
        Cr_up,
        "Cr_reconstructed.png"
    )


    # Save final RGB image
    cv2.imwrite(
        str(output_dir / "reconstructed_RGB.png"),
        cv2.cvtColor(
            image_rec,
            cv2.COLOR_RGB2BGR
        )
    )


    # --------------------------------------
    # Coefficient Analysis
    # --------------------------------------

    Y_stats = analyze_coefficients(
        Y_dct,
        Y_quant
    )

    Cb_stats = analyze_coefficients(
        Cb_dct,
        Cb_quant
    )

    Cr_stats = analyze_coefficients(
        Cr_dct,
        Cr_quant
    )

    total_coefficients = (
        Y_stats["total"] +
        Cb_stats["total"] +
        Cr_stats["total"]
    )

    total_zeroed = (
        Y_stats["zero_after"] +
        Cb_stats["zero_after"] +
        Cr_stats["zero_after"]
    )

    total_newly_zeroed = (
        Y_stats["newly_zeroed"] +
        Cb_stats["newly_zeroed"] +
        Cr_stats["newly_zeroed"]
    )

    zero_percentage = (
        total_zeroed / total_coefficients
    ) * 100

    newly_zeroed_percentage = (
        total_newly_zeroed / total_coefficients
    ) * 100


    print("\n================================")
    print("COEFFICIENT ANALYSIS")
    print("================================")

    for name, stats in [
        ("Y", Y_stats),
        ("Cb", Cb_stats),
        ("Cr", Cr_stats)
    ]:

        print(f"\nChannel: {name}")

        print("Total coefficients:", stats["total"])

        print(
            "Zero after Quantization:",
            stats["zero_after"]
        )

        print(
            f"Zero percentage: "
            f"{stats['zero_percentage']:.2f}%"
        )

        print(
            "Newly zeroed:",
            stats["newly_zeroed"]
        )

    print("\nTotal coefficients:", total_coefficients)

    print("Total zero coefficients:", total_zeroed)

    print(f"Total zero percentage: {zero_percentage:.2f}%")

    print("Newly zeroed coefficients:", total_newly_zeroed)

    print(
        f"Newly zeroed percentage: "
        f"{newly_zeroed_percentage:.2f}%"
    )


    # --------------------------------------
    # Theoretical Data Size
    # --------------------------------------

    size_info = calculate_theoretical_size(
        Y_quant,
        Cb_quant,
        Cr_quant
    )
    
    # --------------------------------------
    # Stage 5: Data Size Comparison
    # --------------------------------------

    stage5_results = calculate_stage5_size(
        original_size,
        Y_dct,
        Cb_dct,
        Cr_dct,
        size_info["estimated_size"]
    )

    # Plot Stage 5 Results
    plot_stage5_results(stage5_results)

    original_size = image_rgb.nbytes

    size_change = (
        (
            size_info["estimated_size"] -
            original_size
        ) / original_size
    ) * 100


    print("\n================================")
    print("THEORETICAL SIZE ANALYSIS")
    print("================================")

    print(
        "Total coefficients:",
        size_info["total_coefficients"]
    )

    print(
        "Non-zero coefficients:",
        size_info["nonzero_coefficients"]
    )

    print(
        "Zero coefficients:",
        size_info["zero_coefficients"]
    )

    print("Original RGB size:", original_size, "Bytes")

    print(
        "Quantized data size:",
        size_info["estimated_size"],
        "Bytes"
    )

    print(
        f"Quantized data size: "
        f"{size_info['size_kb']:.2f} KB"
    )

    print(f"Size change: {size_change:.2f}%")


    # --------------------------------------
    # Compare Stage 3 and Stage 4
    # --------------------------------------

    print("\n================================")
    print("STAGE 3 vs STAGE 4")
    print("================================")

    print(
        f"Stage 3 (N=64) PSNR: "
        f"{psnr_stage3:.4f} dB"
    )

    print(
        f"Stage 4 (JPEG) PSNR: "
        f"{psnr:.4f} dB"
    )

    print(
        f"PSNR Difference: "
        f"{psnr_stage3 - psnr:.4f} dB"
    )


    # --------------------------------------
    # Display All Results
    # --------------------------------------

    display_results(
        image_rgb,
        Y,
        Cb_sub,
        Cr_sub,
        Y_rec,
        Cb_up,
        Cr_up,
        image_rec,
        psnr
    )


    print("\n================================")
    print("Stage 4 Completed Successfully!")
    print("Output Directory:", output_dir)
    print("================================")



if __name__ == "__main__":
    main()
