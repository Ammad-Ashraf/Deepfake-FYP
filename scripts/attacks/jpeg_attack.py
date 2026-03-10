# =============================================================================
# attacks/jpeg_attack.py — JPEG Compression Statistical Attack
# =============================================================================
# CONCEPT — What is this attack doing?
#
#   Deepfake detectors work by finding tiny artifacts that GAN networks leave
#   behind in generated images. These artifacts exist in the HIGH-FREQUENCY
#   components of an image — the fine pixel-level details.
#
#   JPEG compression works by DISCARDING high-frequency information.
#   It's the same reason a photo looks blocky/blurry when you save it at
#   very low quality — the fine details have been thrown away.
#
#   By aggressively compressing each frame, we throw away the exact details
#   the detector is looking for — it's like shredding the forensic evidence.
#
# REAL-WORLD RELEVANCE:
#   This is the MOST realistic attack because it mimics what naturally happens
#   to any video shared online. WhatsApp compresses videos to ~70% quality.
#   Twitter, Instagram, and Telegram all apply their own compression.
#   A deepfake that survives JPEG compression is genuinely dangerous.
#
# WHAT "QUALITY" MEANS:
#   quality=100 → No compression, original quality
#   quality=85  → What most websites use (barely noticeable)
#   quality=50  → Noticeable degradation
#   quality=15  → Heavy compression — visible blockiness, but still watchable
#   quality=5   → Extreme — image becomes heavily pixelated
#   We use quality=15 for an aggressive but realistic attack
#
# LIBRARIES USED:
#   cv2 (OpenCV) — Handles JPEG encoding/decoding via imencode/imdecode
#   numpy        — Frames are numpy arrays; buffer operations use numpy
# =============================================================================

import cv2
import numpy as np
import logging

logger = logging.getLogger("jpeg_attack")


def apply_jpeg_compression(frame: np.ndarray, quality: int = 15) -> np.ndarray:
    """
    Applies JPEG compression to a single video frame.

    The trick here is we DON'T save the frame to disk.
    Instead we:
      1. Encode the frame to JPEG format IN MEMORY (as a byte buffer)
      2. Immediately decode it back to a numpy array
    This round-trip through JPEG encoding/decoding applies all the
    compression artifacts without creating temporary files.
    This is much faster than writing to disk and reading back.

    Args:
        frame   : Single video frame as a numpy array (H x W x 3, BGR format)
                  OpenCV uses BGR (Blue-Green-Red) not RGB — important to remember
        quality : JPEG quality level (1-100, lower = more compression)

    Returns:
        Compressed frame as numpy array, same shape as input
    """

    # --- Validate input ---
    if frame is None or frame.size == 0:
        logger.warning("Received empty frame — returning as-is")
        return frame

    # Clamp quality to valid JPEG range
    quality = max(1, min(100, quality))

    # --- JPEG encode to memory buffer ---
    # cv2.imencode converts a numpy frame to a compressed image byte buffer
    # Parameters:
    #   ".jpg"                      → format to encode as
    #   frame                       → the numpy array to encode
    #   [cv2.IMWRITE_JPEG_QUALITY, quality] → encoding parameters
    #
    # Returns:
    #   success (bool) — True if encoding worked
    #   buffer (numpy array of uint8 bytes) — the compressed JPEG data in memory
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    if not success:
        logger.warning("JPEG encoding failed — returning original frame")
        return frame

    # --- Decode back to numpy array ---
    # cv2.imdecode converts the byte buffer back to a numpy image array
    # cv2.IMREAD_COLOR means read as a 3-channel BGR color image
    compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if compressed_frame is None:
        logger.warning("JPEG decoding failed — returning original frame")
        return frame

    return compressed_frame


# =============================================================================
# BATCH FUNCTION: apply to a list of frame paths
# =============================================================================
def attack_frames(frame_paths: list, quality: int = 15) -> list:
    """
    Applies JPEG compression to every frame in a list.
    Modifies frames IN-PLACE by overwriting the .png files.

    This is what pipeline.py calls — it passes in the list of extracted
    frame paths, and this function compresses each one and saves it back.

    Args:
        frame_paths : List of paths to .png frame files
        quality     : JPEG quality level to apply

    Returns:
        Same list of frame_paths (files have been modified on disk)
    """

    attacked_count = 0
    failed_count = 0

    for frame_path in frame_paths:
        # Read the original clean frame from disk
        frame = cv2.imread(frame_path)

        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            failed_count += 1
            continue

        # Apply the JPEG compression attack
        attacked_frame = apply_jpeg_compression(frame, quality=quality)

        # Save the attacked frame BACK to the same path
        # This overwrites the original clean frame with the compressed version
        # We still save as PNG (lossless) to avoid double-compression effects
        cv2.imwrite(frame_path, attacked_frame)
        attacked_count += 1

    logger.info(f"JPEG attack complete: {attacked_count} frames attacked, "
                f"{failed_count} failed")

    return frame_paths


# =============================================================================
# STANDALONE TEST — Run this file directly to test the attack on one frame
# Usage: python attacks/jpeg_attack.py
# =============================================================================
if __name__ == "__main__":
    import sys

    print("JPEG Compression Attack — Test Mode")
    print("Creating a synthetic test frame (white noise image)...")

    # Create a fake test frame: 256x256 pixels of random color noise
    # This simulates what a video frame looks like as a numpy array
    test_frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Apply at different quality levels to see the effect
    for q in [85, 50, 15, 5]:
        attacked = apply_jpeg_compression(test_frame, quality=q)
        # Calculate mean difference to show how much data was changed
        diff = np.mean(np.abs(test_frame.astype(float) - attacked.astype(float)))
        print(f"  Quality={q:3d} → Mean pixel difference: {diff:.2f}")

    print("Test complete. JPEG attack is working correctly.")
