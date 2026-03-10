# =============================================================================
# config.py — Central Configuration File for Statistical Attack Pipeline
# =============================================================================
# PURPOSE:
#   This is the SINGLE source of truth for all parameters in the project.
#   Instead of hardcoding numbers inside individual scripts, everything
#   tunable lives here. This means:
#     - You change attack intensity in ONE place, it updates everywhere
#     - Your teammates can read this file to understand what values you used
#     - When writing the FYP report, you can cite exact parameters
#
# HOW TO USE:
#   In any other script, just write:
#       from config import CONFIG
#   Then access values like:
#       CONFIG["jpeg"]["quality"]
# =============================================================================

import os

CONFIG = {

    # -------------------------------------------------------------------------
    # PATHS — All folder locations relative to the project root
    # -------------------------------------------------------------------------
    "paths": {
        # Where your original FakeAVCeleb fake video samples live
        # UPDATE THIS when your dataset finishes downloading
        # Example: "C:/Users/Ammad/Downloads/FakeAVCeleb/FakeVideo-RealAudio"
        "input_dir": "./input",

        # Root folder where all attacked videos will be saved
        "output_dir": "./output",

        # Sub-folders inside output — one per attack type
        # These match the folder names we already created
        "output_subfolders": {
            "jpeg":      "./output/jpeg_compression",
            "gaussian":  "./output/gaussian_noise",
            "dct":       "./output/dct_manipulation",
            "histogram": "./output/histogram_shift",
        },

        # Temporary folder — used during processing to hold extracted frames
        # Gets deleted automatically after each video is processed
        "temp_dir": "./temp",

        # Logs folder — stores progress tracking and error logs
        "logs_dir": "./logs",

        # The JSON file that tracks which videos are done/pending
        # This is what makes the pipeline RESUMABLE
        "progress_file": "./logs/progress.json",

        # Plain text log — records timestamps, errors, and skipped files
        "log_file": "./logs/attack_log.txt",
    },

    # -------------------------------------------------------------------------
    # VIDEO PROCESSING — How frames are extracted and reconstructed
    # -------------------------------------------------------------------------
    "video": {
        # Frames Per Second to extract from the video
        # FakeAVCeleb videos are typically 30fps
        # We use 30 here to process every frame — no skipping
        "fps": 30,

        # Output video codec
        # "mp4v" is the standard codec for .mp4 files using OpenCV
        # It's widely supported and doesn't require extra libraries
        "codec": "mp4v",

        # Maximum width/height of frames (resize if larger to save memory)
        # FakeAVCeleb frames are typically 256x256 or 224x224
        # Set to None if you don't want any resizing
        "max_frame_size": None,
    },

    # -------------------------------------------------------------------------
    # ATTACK 1: JPEG COMPRESSION
    # -------------------------------------------------------------------------
    # CONCEPT:
    #   JPEG compression works by discarding high-frequency details in images.
    #   Deepfake detectors rely on those tiny pixel artifacts left by GAN generation.
    #   By heavily compressing frames, we DESTROY those artifacts — the detector
    #   can no longer find its evidence. This is the most realistic attack because
    #   all media shared on WhatsApp, Twitter, etc. gets compressed automatically.
    #
    # PARAMETER EXPLANATION:
    #   quality=15 means very heavy compression (0=worst, 100=best)
    #   quality=85 is what most platforms use — barely noticeable
    #   We use quality=15 to be aggressive — this is an ATTACK, not archiving
    # -------------------------------------------------------------------------
    "jpeg": {
        "quality": 15,          # Compression strength (lower = more aggressive)
        "encode_param": [       # OpenCV encoding parameter format
            int(cv2_jpeg_quality := 15)  # Will be set properly in attack script
        ],
    },

    # -------------------------------------------------------------------------
    # ATTACK 2: GAUSSIAN NOISE
    # -------------------------------------------------------------------------
    # CONCEPT:
    #   Gaussian noise adds random pixel variations drawn from a normal (bell curve)
    #   distribution. Real photos have natural sensor noise — GANs don't.
    #   Detectors can recognize the ABSENCE of natural noise as a forgery signal.
    #   By injecting calibrated Gaussian noise, we make fake frames look more
    #   like real camera captures, fooling the statistical fingerprint analysis.
    #
    # PARAMETER EXPLANATION:
    #   mean=0 means noise is centered at zero (no overall brightness shift)
    #   sigma=25 controls how strong the noise is (higher = more visible)
    #   25 is aggressive but still looks natural — like a low-light photo
    #   For a subtler attack use sigma=10; for stronger use sigma=35
    # -------------------------------------------------------------------------
    "gaussian": {
        "mean": 0,      # Center of the noise distribution (don't change this)
        "sigma": 25,    # Spread/strength of the noise — KEY PARAMETER
    },

    # -------------------------------------------------------------------------
    # ATTACK 3: DCT MANIPULATION (Discrete Cosine Transform)
    # -------------------------------------------------------------------------
    # CONCEPT:
    #   DCT is the mathematical foundation of JPEG compression. It converts
    #   an image from pixel space into FREQUENCY space — think of it like
    #   breaking a song into its individual musical notes.
    #   High frequency components = fine details and edges
    #   Low frequency components = broad shapes and colors
    #
    #   GAN-generated images leave characteristic patterns in the high-frequency
    #   DCT coefficients — this is a known forensic fingerprint.
    #   Our attack zeros out (suppresses) those high-frequency components,
    #   erasing the GAN's digital footprint in frequency space.
    #
    # PARAMETER EXPLANATION:
    #   block_size=8 → we process 8x8 pixel blocks (same as JPEG standard)
    #   keep_fraction=0.2 → we KEEP only the lowest 20% of frequency components
    #                        and zero out the rest (the high-freq GAN artifacts)
    # -------------------------------------------------------------------------
    "dct": {
        "block_size": 8,        # Size of pixel blocks to transform (8 is standard)
        "keep_fraction": 0.2,   # Fraction of frequencies to KEEP (0.2 = bottom 20%)
    },

    # -------------------------------------------------------------------------
    # ATTACK 4: COLOR HISTOGRAM SHIFTING
    # -------------------------------------------------------------------------
    # CONCEPT:
    #   A color histogram shows how pixel values are distributed across an image
    #   (how many dark pixels, how many bright pixels, etc.).
    #   GANs produce images with slightly different color distributions than
    #   real cameras — this is a detectable statistical signature.
    #   Our attack shifts these distributions by a small random amount,
    #   disguising the GAN's characteristic color patterns.
    #
    # PARAMETER EXPLANATION:
    #   shift_range=[-30, 30] → each color channel (R,G,B) gets shifted by a
    #                           random value between -30 and +30 pixel intensity
    #   We clip values to [0,255] after shifting so nothing overflows
    # -------------------------------------------------------------------------
    "histogram": {
        "shift_range": (-30, 30),   # Min and max shift per color channel
        "clip_min": 0,              # Minimum pixel value (don't go below black)
        "clip_max": 255,            # Maximum pixel value (don't go above white)
    },

    # -------------------------------------------------------------------------
    # PIPELINE BEHAVIOR — How the orchestrator manages the process
    # -------------------------------------------------------------------------
    "pipeline": {
        # If True, divide the input videos across the 4 attacks (e.g., 25% each)
        # instead of running every attack on every video. Saves 75% processing time
        # and disk space, which is critical when running on CPU without GPU.
        "split_dataset": True,

        # How many frames to process before saving a checkpoint to progress.json
        # Lower = safer against crashes but slightly slower
        "checkpoint_every_n_videos": 5,

        # If True, skip videos that already exist in output (resumability)
        # Set to False ONLY if you want to reprocess everything from scratch
        "skip_completed": True,

        # If True, print detailed frame-level logs (very verbose)
        # Set to False for cleaner output during long runs
        "verbose": False,

        # File extensions to look for in the input directory
        "valid_extensions": [".mp4", ".MP4"],
    },
}


# =============================================================================
# HELPER: Print a summary of current config when this file is run directly
# This is useful for quickly checking your settings before a long run
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DeepTrust — Statistical Attack Pipeline Configuration")
    print("=" * 60)
    print(f"  Input  directory : {CONFIG['paths']['input_dir']}")
    print(f"  Output directory : {CONFIG['paths']['output_dir']}")
    print(f"  Temp   directory : {CONFIG['paths']['temp_dir']}")
    print()
    print("  Attack Parameters:")
    print(f"    JPEG quality       : {CONFIG['jpeg']['quality']}")
    print(f"    Gaussian sigma     : {CONFIG['gaussian']['sigma']}")
    print(f"    DCT keep fraction  : {CONFIG['dct']['keep_fraction']}")
    print(f"    Histogram shift    : {CONFIG['histogram']['shift_range']}")
    print("=" * 60)
