# =============================================================================
# attacks/histogram_attack.py — Color Histogram Shifting Attack
# =============================================================================
# CONCEPT — What is this attack doing?
#
#   WHAT IS A COLOR HISTOGRAM?
#   ──────────────────────────
#   A color histogram shows HOW PIXEL VALUES ARE DISTRIBUTED in an image.
#   For a grayscale image: it shows how many pixels are dark, medium, bright.
#   For a color image: there are 3 histograms — one each for Blue, Green, Red.
#
#   Example: If a face photo has a Red histogram peak at value 180,
#   that means most pixels in the red channel have intensity ≈ 180.
#
#   GAN FINGERPRINTS IN HISTOGRAMS:
#   ───────────────────────────────
#   GAN-generated images have characteristic statistical distributions.
#   Different GAN architectures (StyleGAN2, FaceSwap, etc.) each produce
#   slightly different color distribution "fingerprints".
#
#   Some deepfake detectors compare the histogram of a suspected fake against
#   known GAN distribution patterns. If the color statistics match a known
#   GAN fingerprint → flagged as fake.
#
#   THE ATTACK:
#   ───────────
#   We SHIFT the color values in each channel by a random amount.
#   If the GAN's fingerprint has a Red channel peak at 180,
#   and we shift Red by +20, the peak moves to 200.
#   The detector no longer finds a match at 180 → fingerprint erased.
#
#   Each channel (B, G, R) gets an INDEPENDENT random shift.
#   This changes the overall color tint of the frame slightly
#   but preserves all the visual content (faces, backgrounds, etc.)
#
# VISIBLE EFFECT:
#   The shift is subtle. shift_range=(-30, 30) means the image might get
#   very slightly warmer, cooler, or tinted. Not enough to notice casually,
#   but enough to disrupt statistical fingerprint matching.
#
# WHY THIS IS INTERESTING ACADEMICALLY:
#   This attack is essentially a form of "domain shift" — we're moving the
#   data out of the distribution the detector was trained on. This is related
#   to why detectors fail on different datasets (see the 45-50% real-world
#   accuracy problem in our problem statement).
#
# LIBRARIES USED:
#   numpy — All operations are array math (addition, clipping, random)
#   cv2   — Reading/writing frames
# =============================================================================

import cv2
import numpy as np
import logging

logger = logging.getLogger("histogram_attack")


def apply_histogram_shift(
    frame: np.ndarray,
    shift_range: tuple = (-30, 30),
    clip_min: int = 0,
    clip_max: int = 255,
    seed: int = None
) -> np.ndarray:
    """
    Randomly shifts the color distribution of each channel in a video frame.

    The process:
      1. For each of the 3 color channels (B, G, R):
         a. Draw a random shift value from shift_range
         b. Add that shift to all pixel values in the channel
         c. Clip values to [clip_min, clip_max] to prevent overflow
      2. Reconstruct the frame from the 3 modified channels

    Args:
        frame       : Video frame as numpy array (H x W x 3, BGR, uint8)
        shift_range : Tuple (min_shift, max_shift) — range for random shifts
                      Each channel gets an INDEPENDENT random value from this range
        clip_min    : Minimum pixel value after shift (prevents going below 0)
        clip_max    : Maximum pixel value after shift (prevents going above 255)
        seed        : Random seed for reproducibility (None = fully random)
                      Set a fixed seed if you need identical results across runs

    Returns:
        Color-shifted frame as numpy array, same shape as input
    """

    if frame is None or frame.size == 0:
        logger.warning("Received empty frame — returning as-is")
        return frame

    # Optional: set random seed for reproducibility
    # In production attacks we DON'T set a seed — randomness is the point
    if seed is not None:
        np.random.seed(seed)

    # Convert to int32 for the addition step
    # We need int32 (not uint8) because:
    #   - uint8 range is 0-255
    #   - If pixel=10 and shift=-30, result would be -20
    #   - uint8 can't hold negative numbers — it wraps around to 236 (wrong!)
    #   - int32 can hold the intermediate negative values before clipping
    frame_int = frame.astype(np.int32)

    # Process each channel independently
    channels = cv2.split(frame_int)
    # cv2.split() decomposes (H x W x 3) array into 3 separate (H x W) arrays
    # channels[0] = Blue, channels[1] = Green, channels[2] = Red  (BGR order)

    shifted_channels = []
    shift_values = []  # Track for logging

    for channel in channels:
        # Draw a random integer shift from the specified range
        # np.random.randint(low, high) returns a random integer in [low, high)
        shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        shift_values.append(shift)

        # Add the scalar shift to every pixel in this channel
        # Broadcasting: adding a single number to a 2D array adds it everywhere
        shifted = channel + shift

        # Clip to prevent pixel overflow/underflow
        # np.clip ensures every value stays in [clip_min, clip_max]
        shifted = np.clip(shifted, clip_min, clip_max)

        shifted_channels.append(shifted)

    logger.debug(f"Channel shifts — B:{shift_values[0]:+d} G:{shift_values[1]:+d} "
                 f"R:{shift_values[2]:+d}")

    # Merge channels back together and convert to uint8
    result = cv2.merge(shifted_channels)
    # cv2.merge() recombines 3 (H x W) arrays back into one (H x W x 3) array

    return result.astype(np.uint8)


# =============================================================================
# BATCH FUNCTION: apply to a list of frame paths
# =============================================================================
def attack_frames(
    frame_paths: list,
    shift_range: tuple = (-30, 30),
    clip_min: int = 0,
    clip_max: int = 255
) -> list:
    """
    Applies color histogram shifting to every frame in a list.

    IMPORTANT DESIGN DECISION — PER-FRAME vs PER-VIDEO SHIFTS:
    We generate a NEW random shift for EACH frame independently.
    Alternative would be one shift per video (same shift applied to all frames).

    We chose per-frame because:
    1. It breaks temporal consistency — harder for detectors to average out
    2. It's more similar to real compression variability
    3. Creates a more challenging attack dataset for hardening in Iteration 3

    Args:
        frame_paths : List of paths to .png frame files
        shift_range : Range for per-channel random color shifts
        clip_min    : Minimum pixel value
        clip_max    : Maximum pixel value

    Returns:
        Same list of frame_paths (files modified on disk)
    """

    attacked_count = 0
    failed_count = 0

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)

        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            failed_count += 1
            continue

        attacked_frame = apply_histogram_shift(
            frame,
            shift_range=shift_range,
            clip_min=clip_min,
            clip_max=clip_max
        )

        cv2.imwrite(frame_path, attacked_frame)
        attacked_count += 1

    logger.info(f"Histogram attack complete: {attacked_count} frames attacked, "
                f"{failed_count} failed")

    return frame_paths


# =============================================================================
# STANDALONE TEST
# Usage: python attacks/histogram_attack.py
# =============================================================================
if __name__ == "__main__":
    print("Color Histogram Shift Attack — Test Mode")

    # Create a uniform test frame (all pixels = 128)
    # Easy to verify: shifting by 30 should give all pixels = 158
    test_frame = np.full((256, 256, 3), 128, dtype=np.uint8)

    print(f"Original frame — mean per channel: "
          f"B={test_frame[:,:,0].mean():.0f} "
          f"G={test_frame[:,:,1].mean():.0f} "
          f"R={test_frame[:,:,2].mean():.0f}")

    # Test with different ranges
    for shift_range in [(-10, 10), (-30, 30), (-60, 60)]:
        np.random.seed(42)  # Fix seed for reproducible test
        attacked = apply_histogram_shift(test_frame, shift_range=shift_range, seed=42)
        print(f"  Range={shift_range} → "
              f"B={attacked[:,:,0].mean():.0f} "
              f"G={attacked[:,:,1].mean():.0f} "
              f"R={attacked[:,:,2].mean():.0f}")

    print("Test complete. Histogram attack is working correctly.")
