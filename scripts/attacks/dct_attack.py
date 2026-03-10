# =============================================================================
# attacks/dct_attack.py — DCT (Discrete Cosine Transform) Manipulation Attack
# =============================================================================
# CONCEPT — What is this attack doing?
#
#   UNDERSTANDING DCT (the math made simple):
#   ─────────────────────────────────────────
#   Think of any image as being made up of many different "waves" or patterns
#   overlaid on top of each other, just like how a musical chord is made up
#   of individual notes.
#
#   The DCT is a mathematical operation that DECOMPOSES an image from
#   "pixel space" into "frequency space":
#
#     Pixel space    → You see individual pixel values (e.g., this pixel is 128)
#     Frequency space → You see coefficients representing patterns
#                      Low frequency  = broad strokes, overall shapes, background
#                      High frequency = fine edges, textures, tiny details
#
#   The DCT works on 8x8 pixel blocks (the same size used by JPEG).
#   After the DCT, the top-left corner of each block contains LOW frequencies,
#   and the bottom-right corner contains HIGH frequencies:
#
#     [LF  LF  LF  HF  HF  HF  HF  HF]
#     [LF  LF  MF  HF  HF  HF  HF  HF]
#     [LF  MF  MF  HF  HF  HF  HF  HF]   ← 8x8 DCT block
#     [HF  HF  HF  HF  HF  HF  HF  HF]
#     ...
#
#   THE ATTACK:
#   ──────────
#   GAN networks are known to leave SPECIFIC PATTERNS in the high-frequency
#   DCT coefficients. Researchers have published papers showing that you can
#   literally VISUALIZE GAN fingerprints by looking at the DCT spectrum.
#
#   Our attack zeros out the high-frequency coefficients in every 8x8 block.
#   keep_fraction=0.2 means we keep ONLY the bottom-left 20% (low frequencies)
#   and zero out the rest. This erases the GAN's frequency-domain fingerprint.
#
#   This is MORE SURGICAL than JPEG compression:
#   - JPEG: Discard high frequencies during encoding (lossy side effect)
#   - DCT Attack: DIRECTLY zero out specific frequency bands (intentional attack)
#
# WHY THIS MATTERS FOR ITERATION 3:
#   The adversarial hardening team needs to know that detectors trained to look
#   for frequency-domain artifacts can be defeated. This dataset proves that
#   vulnerability exists.
#
# LIBRARIES USED:
#   numpy       — Matrix operations for the DCT block processing
#   cv2         — Reading/writing frames, color space conversion
#   scipy       — We use scipy.fftpack.dct and idct for the actual transform
#                 Install: pip install scipy
#                 Why scipy over numpy? scipy's dct is FASTER and uses the
#                 standard DCT-II definition (same as JPEG uses)
# =============================================================================

import cv2
import numpy as np
import logging

# scipy.fftpack provides the DCT (Discrete Cosine Transform) function
# This is the same mathematical transform used inside JPEG compression
from scipy.fftpack import dct, idct

logger = logging.getLogger("dct_attack")


def apply_dct_to_block(block: np.ndarray, keep_fraction: float = 0.2) -> np.ndarray:
    """
    Applies DCT manipulation to a single 8x8 pixel block.

    Steps:
      1. Apply 2D DCT to convert the block from pixel to frequency space
      2. Create a mask that keeps only the low-frequency coefficients
      3. Zero out everything else (high-frequency coefficients)
      4. Apply inverse DCT to convert back to pixel space

    Args:
        block         : 8x8 numpy array of pixel values (single color channel)
        keep_fraction : What fraction of coefficients to keep (0.2 = bottom 20%)

    Returns:
        8x8 numpy array of modified pixel values
    """

    # --- Step 1: Forward 2D DCT ---
    # We apply DCT twice: first along rows (axis=0), then along columns (axis=1)
    # This gives us the 2D DCT (standard for image processing)
    # norm='ortho' uses orthonormal normalization — makes the transform reversible
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    # Shape is still 8x8, but values are now frequency coefficients, not pixel values

    # --- Step 2: Create frequency mask ---
    # We want to keep the TOP-LEFT corner (low frequencies) and zero the rest
    # keep_fraction=0.2 of an 8x8 block means we keep roughly the top-left 2x2 area
    block_size = block.shape[0]  # Should be 8

    # Calculate how many rows/columns to keep
    keep_n = max(1, int(block_size * keep_fraction))
    # max(1,...) ensures we always keep at least 1 coefficient

    # Create a zero mask (same size as block)
    mask = np.zeros_like(dct_block)

    # Set the top-left keep_n x keep_n region to 1 (keep these coefficients)
    mask[:keep_n, :keep_n] = 1.0

    # --- Step 3: Apply mask — zero out high frequencies ---
    dct_block_filtered = dct_block * mask
    # Multiplying by 0 zeros out those coefficients
    # Multiplying by 1 keeps the low-frequency coefficients unchanged

    # --- Step 4: Inverse DCT — convert back to pixel space ---
    # We reverse the transform: inverse DCT along columns, then rows
    reconstructed = idct(idct(dct_block_filtered.T, norm='ortho').T, norm='ortho')

    return reconstructed


def apply_dct_manipulation(
    frame: np.ndarray,
    block_size: int = 8,
    keep_fraction: float = 0.2
) -> np.ndarray:
    """
    Applies DCT frequency manipulation to an entire video frame.

    We process the frame in 8x8 blocks (same approach as JPEG).
    For each block in each color channel, we suppress high-frequency components.
    Blocks at the edges of the image that don't fit perfectly are padded and trimmed.

    Args:
        frame         : Video frame as numpy array (H x W x 3, BGR, uint8)
        block_size    : Size of blocks to process (8 is standard)
        keep_fraction : Fraction of DCT coefficients to keep (0.2 = 20%)

    Returns:
        DCT-manipulated frame as numpy array, same shape as input
    """

    if frame is None or frame.size == 0:
        logger.warning("Received empty frame — returning as-is")
        return frame

    # Work with a float copy of the frame
    # DCT values are not integers — we need floating point precision
    frame_float = frame.astype(np.float64)

    height, width, channels = frame_float.shape

    # Process each color channel separately (B, G, R)
    # DCT is applied per-channel, not on RGB triplets
    result = np.zeros_like(frame_float)

    for c in range(channels):
        channel = frame_float[:, :, c]  # Extract single color channel (2D array)

        # Process the channel in block_size x block_size blocks
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):

                # Extract current block
                # If block extends beyond image border, it gets naturally truncated
                block = channel[row:row+block_size, col:col+block_size]

                # Only process full-sized blocks to avoid DCT artifacts at edges
                if block.shape[0] == block_size and block.shape[1] == block_size:
                    modified_block = apply_dct_to_block(block, keep_fraction)
                    result[row:row+block_size, col:col+block_size, c] = modified_block
                else:
                    # Edge blocks: keep them unchanged (they're usually tiny slivers)
                    result[row:row+block_size, col:col+block_size, c] = block

    # Clip and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# =============================================================================
# BATCH FUNCTION: apply to a list of frame paths
# =============================================================================
def attack_frames(
    frame_paths: list,
    block_size: int = 8,
    keep_fraction: float = 0.2
) -> list:
    """
    Applies DCT manipulation to every frame in a list.

    NOTE ON PERFORMANCE:
    DCT is the most computationally expensive of the 4 attacks because we
    process every 8x8 block individually. For a 256x256 frame, that's
    (256/8) * (256/8) * 3 channels = 3072 DCT operations per frame.
    On CPU this will be slower than JPEG or Gaussian — be patient.

    Args:
        frame_paths   : List of paths to .png frame files
        block_size    : DCT block size (8 is standard)
        keep_fraction : Fraction of frequencies to retain

    Returns:
        Same list of frame_paths (files modified on disk)
    """

    attacked_count = 0
    failed_count = 0

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)

        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            failed_count += 1
            continue

        attacked_frame = apply_dct_manipulation(
            frame,
            block_size=block_size,
            keep_fraction=keep_fraction
        )

        cv2.imwrite(frame_path, attacked_frame)
        attacked_count += 1

        # Log progress every 50 frames (DCT is slow, good to see it moving)
        if (i + 1) % 50 == 0:
            logger.info(f"DCT: processed {i+1}/{len(frame_paths)} frames")

    logger.info(f"DCT attack complete: {attacked_count} frames attacked, "
                f"{failed_count} failed")

    return frame_paths


# =============================================================================
# STANDALONE TEST
# Usage: python attacks/dct_attack.py
# =============================================================================
if __name__ == "__main__":
    print("DCT Manipulation Attack — Test Mode")

    # Create a simple test block with a recognizable pattern
    test_block = np.array([
        [10, 20, 30, 40, 50, 60, 70, 80],
        [15, 25, 35, 45, 55, 65, 75, 85],
        [20, 30, 40, 50, 60, 70, 80, 90],
        [25, 35, 45, 55, 65, 75, 85, 95],
        [30, 40, 50, 60, 70, 80, 90, 100],
        [35, 45, 55, 65, 75, 85, 95, 105],
        [40, 50, 60, 70, 80, 90, 100, 110],
        [45, 55, 65, 75, 85, 95, 105, 115],
    ], dtype=np.float64)

    print(f"Original block[0,0]: {test_block[0,0]:.1f}")

    for keep in [0.5, 0.2, 0.1]:
        result = apply_dct_to_block(test_block, keep_fraction=keep)
        diff = np.mean(np.abs(test_block - result))
        print(f"  keep_fraction={keep} → Mean block difference: {diff:.2f}")

    print("Test complete. DCT attack is working correctly.")
