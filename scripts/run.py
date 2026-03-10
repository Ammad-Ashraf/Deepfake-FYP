# =============================================================================
# run.py — Single Entry Point for the Statistical Attack Pipeline
# =============================================================================
# PURPOSE:
#   This is the ONLY file you need to run directly.
#   Everything else (pipeline, attacks, utils) is called from here.
#
# USAGE EXAMPLES:
#
#   Run a single attack type:
#     python run.py --attack gaussian
#     python run.py --attack jpeg
#     python run.py --attack dct
#     python run.py --attack histogram
#
#   Run ALL 4 attack types sequentially (one after another):
#     python run.py --attack all
#
#   Update input path without editing config.py:
#     python run.py --attack gaussian --input "D:/FakeAVCeleb/FakeVideo-RealAudio"
#
#   Run all attacks with custom input path:
#     python run.py --attack all --input "D:/FakeAVCeleb/FakeVideo-RealAudio"
#
#   Test the system with just 10 videos (good for first-time verification):
#     python run.py --attack gaussian --limit 10
#
#   Check your config without running anything:
#     python run.py --config-check
#
# LIBRARY: argparse
#   Built-in Python library for parsing command-line arguments.
#   It automatically generates a --help message too:
#     python run.py --help
# =============================================================================

import argparse
import os
import sys

# Add scripts folder to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from pipeline import run_attack_pipeline


def update_input_path(new_path: str):
    """
    Temporarily override the input path from config.py.
    Useful when running from command line without editing config.

    This updates the CONFIG dict IN MEMORY only — doesn't change config.py.
    Next time you run without --input, it reverts to the config.py value.
    """
    CONFIG["paths"]["input_dir"] = new_path
    print(f"[run.py] Input path overridden to: {new_path}")


def config_check():
    """
    Validates the current configuration before running.
    Checks that:
    - Input directory exists (or warns if not yet set)
    - Output directories can be created
    - Required libraries are installed
    """
    print("\n" + "="*60)
    print("DeepTrust — Configuration Check")
    print("="*60)

    # Check input directory
    input_dir = CONFIG["paths"]["input_dir"]
    if os.path.exists(input_dir):
        # Count video files
        video_count = sum(
            1 for root, dirs, files in os.walk(input_dir)
            for f in files
            if f.endswith(tuple(CONFIG["pipeline"]["valid_extensions"]))
        )
        print(f"✓ Input directory found: {input_dir}")
        print(f"  └── {video_count} video files found")
    else:
        print(f"✗ Input directory NOT found: {input_dir}")
        print(f"  └── Update CONFIG['paths']['input_dir'] in config.py")
        print(f"      after your dataset finishes downloading")

    # Check output directories
    for attack_name, output_path in CONFIG["paths"]["output_subfolders"].items():
        try:
            os.makedirs(output_path, exist_ok=True)
            print(f"✓ Output folder ready: {output_path}")
        except Exception as e:
            print(f"✗ Cannot create output folder {output_path}: {e}")

    # Check required libraries
    print("\n  Library Check:")
    libraries = {
        "cv2":           "opencv-python",
        "numpy":         "numpy",
        "scipy":         "scipy",
        "tqdm":          "tqdm",
    }

    for lib_name, install_name in libraries.items():
        try:
            __import__(lib_name)
            print(f"  ✓ {lib_name}")
        except ImportError:
            print(f"  ✗ {lib_name} — install with: pip install {install_name}")

    # Check FFmpeg
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print(f"  ✓ ffmpeg")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  ✗ ffmpeg — download from https://ffmpeg.org/download.html")
        print(f"           and add to your system PATH")

    print("="*60 + "\n")


def main():
    # ==========================================================================
    # ARGUMENT PARSER SETUP
    # ==========================================================================
    parser = argparse.ArgumentParser(
        description="DeepTrust — Statistical Attack Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --attack gaussian
  python run.py --attack all
  python run.py --attack jpeg --input "D:/FakeAVCeleb/FakeVideo-RealAudio"
  python run.py --attack all --limit 10
  python run.py --config-check
        """
    )

    # --attack argument: which attack type to run
    parser.add_argument(
        "--attack",
        type=str,
        choices=["jpeg", "gaussian", "dct", "histogram", "all"],
        help="Attack type to apply. Use 'all' to run all 4 sequentially."
    )

    # --input argument: override input directory from config
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to FakeAVCeleb fake video samples folder. "
             "Overrides config.py setting."
    )

    # --limit argument: process only N videos (useful for testing)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N videos. Good for testing the pipeline."
    )

    # --config-check argument: validate setup without running
    parser.add_argument(
        "--config-check",
        action="store_true",
        help="Check configuration and library setup without processing any videos."
    )

    args = parser.parse_args()

    # ==========================================================================
    # HANDLE ARGUMENTS
    # ==========================================================================

    # Config check mode — just validate and exit
    if args.config_check:
        config_check()
        return

    # Require --attack if not doing config-check
    if args.attack is None:
        parser.print_help()
        print("\nError: --attack is required. Use --config-check to validate setup.")
        sys.exit(1)

    # Override input path if provided
    if args.input:
        update_input_path(args.input)

    # Apply limit if provided (for testing)
    if args.limit:
        print(f"\n[run.py] LIMIT MODE: Processing only first {args.limit} videos")
        # We'll pass this into config so pipeline.py can respect it
        CONFIG["pipeline"]["limit"] = args.limit

    # ==========================================================================
    # RUN THE PIPELINE
    # ==========================================================================

    print("\n" + "="*60)
    print("DeepTrust — Statistical Attack Pipeline")
    print("="*60)

    if args.attack == "all":
        # Run all 4 attack types sequentially
        attack_types = ["jpeg", "gaussian", "dct", "histogram"]
        print(f"Running ALL attacks: {' → '.join(attack_types)}")
        print("="*60 + "\n")

        for attack_type in attack_types:
            print(f"\n{'─'*40}")
            print(f"Starting: {attack_type.upper()} attack")
            print(f"{'─'*40}")
            run_attack_pipeline(attack_type)

        print("\n" + "="*60)
        print("ALL ATTACKS COMPLETE")
        print("="*60)

    else:
        # Run a single attack type
        print(f"Running: {args.attack.upper()} attack")
        print("="*60 + "\n")
        run_attack_pipeline(args.attack)


# =============================================================================
# ENTRY POINT
# =============================================================================
# This block only runs when you execute this file directly (python run.py)
# NOT when another file imports it (import run)
if __name__ == "__main__":
    main()
