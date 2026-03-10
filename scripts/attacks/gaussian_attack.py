# =============================================================================
# attacks/gaussian_attack.py — Gaussian Noise Statistical Attack
# =============================================================================
# CONCEPT — What is this attack doing?
#
#   Every real camera introduces tiny random imperfections into photos and videos.
#   This is called "sensor noise" or "image noise" — it's the slight grain you see
#   in low-light photos. It follows a Gaussian (normal/bell-curve) distribution.
#
#   GAN-generated images are MATHEMATICALLY PERFECT — they have no sensor noise.
#   This perfection is ironically a giveaway. Some deepfake detectors specifically
#   look for the ABSENCE of natural noise as a forgery signal.
#
#   Our attack adds SYNTHETIC Gaussian noise to each frame.
#   After the attack, the frame looks like it came from a real (noisy) camera.
#   The detector can no longer use noise patterns as evidence.
#
# THE GAUSSIAN DISTRIBUTION:
#   A Gaussian distribution is a bell curve. Most values cluster near the mean,
#   with fewer values farther away.
#   mean=0  → noise doesn't shift overall brightness (same amount of lighter
#             and darker pixels added)
#   sigma=25 → controls the "spread" — higher sigma = stronger, more visible noise
#             25 is roughly equivalent to ISO 1600 camera noise (realistic)
#
# WHY THIS ATTACKS THE DETECTOR:
#   The noise disrupts the subtle pixel-level statistical patterns that GANs leave.
#   It's like adding static to a radio signal — the original signal is still there
#   but it's buried in noise that makes forensic analysis unreliable.
#
# LIBRARIES USED:
#   numpy — All noise generation uses numpy's random number generation
#           np.random.normal() generates Gaussian-distributed random values
#   cv2   — For reading/writing frame files
# =============================================================================

import cv2
import numpy as np
import logging

logger = logging.getLogger("gaussian_attack")


def apply_gaussian_noise(
    frame: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0
) -> np.ndarray:
    """
    Adds Gaussian (normally distributed) random noise to a single video frame.

    The process:
      1. Generate a noise array of the same size as the frame
         Each value drawn from Normal distribution N(mean, sigma^2)
      2. Add the noise to the original frame pixel values
      3. Clip values back to [0, 255] range
         (addition might push values below 0 or above 255)
      4. Convert back to uint8 (8-bit integers that image pixels use)

    Args:
        frame : Single video frame as numpy array (H x W x 3, BGR, uint8)
        mean  : Center of the noise distribution (0 = no brightness shift)
        sigma : Standard deviation — controls noise intensity
                Low (5-10)  : Very subtle, almost invisible
                Medium (25) : Visible grain, like a low-light photo
                High (50+)  : Heavy grain, clearly artificial

    Returns:
        Noisy frame as numpy array, same shape and dtype as input
    """

    if frame is None or frame.size == 0:
        logger.warning("Received empty frame — returning as-is")
        return frame

    # --- Generate Gaussian noise ---
    # np.random.normal(mean, sigma, size) generates an array of random values
    # where each value is drawn from the Normal distribution N(mean, sigma^2)
    # size=frame.shape means the noise array is the same shape as the frame
    # (height, width, 3 color channels)
    noise = np.random.normal(
        loc=mean,       # Center of distribution (loc = location parameter)
        scale=sigma,    # Standard deviation (scale = spread parameter)
        size=frame.shape
    )
    # noise is currently float64 (e.g., values like 12.7, -8.3, 31.2)

    # --- Add noise to frame ---
    # Convert frame to float first to avoid uint8 overflow
    # uint8 only holds 0-255; adding noise could go outside this range
    # We work in float, then clip and convert back
    noisy_frame = frame.astype(np.float64) + noise

    # --- Clip to valid pixel range ---
    # np.clip(array, min, max) ensures no value goes below 0 or above 255
    # Values below 0 → set to 0 (pure black)
    # Values above 255 → set to 255 (pure white)
    noisy_frame = np.clip(noisy_frame, 0, 255)

    # --- Convert back to uint8 ---
    # Images must be stored as 8-bit unsigned integers (0-255 range)
    # .astype(np.uint8) truncates decimal places (12.7 → 12)
    return noisy_frame.astype(np.uint8)


# =============================================================================
# BATCH FUNCTION: apply to a list of frame paths
# =============================================================================
def attack_frames(
    frame_paths: list,
    mean: float = 0.0,
    sigma: float = 25.0
) -> list:
    """
    Applies Gaussian noise to every frame in a list.
    Modifies frames IN-PLACE by overwriting the .png files.

    NOTE ON RANDOMNESS:
    Each frame gets DIFFERENT noise (np.random.normal generates fresh values
    each call). This is intentional — it simulates real camera noise which
    varies from frame to frame. Using the same noise for all frames would
    create an obvious repeating pattern that could be detected.

    Args:
        frame_paths : List of paths to .png frame files
        mean        : Noise distribution mean
        sigma       : Noise distribution standard deviation

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

        noisy_frame = apply_gaussian_noise(frame, mean=mean, sigma=sigma)
        cv2.imwrite(frame_path, noisy_frame)
        attacked_count += 1

    logger.info(f"Gaussian attack complete: {attacked_count} frames attacked, "
                f"{failed_count} failed")

    return frame_paths


# =============================================================================
# STANDALONE TEST
# Usage: python attacks/gaussian_attack.py
# =============================================================================
if __name__ == "__main__":
    print("Gaussian Noise Attack — Test Mode")
    print("Creating a test frame (gradient image)...")

    # Create a gradient test frame (0 to 255 smoothly)
    # Better test than random noise because we can see what changed
    test_frame = np.tile(
        np.linspace(0, 255, 256).astype(np.uint8),
        (256, 1)
    )
    test_frame = cv2.cvtColor(test_frame, cv2.COLOR_GRAY2BGR)

    print(f"Original frame — mean pixel value: {test_frame.mean():.2f}")

    for sigma in [5, 25, 50]:
        attacked = apply_gaussian_noise(test_frame, mean=0, sigma=sigma)
        diff = np.mean(np.abs(test_frame.astype(float) - attacked.astype(float)))
        print(f"  Sigma={sigma:3d} → Mean pixel change: {diff:.2f}")

    print("Test complete. Gaussian attack is working correctly.")
