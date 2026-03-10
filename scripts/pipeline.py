# =============================================================================
# pipeline.py — Main Orchestrator for the Statistical Attack Pipeline
# =============================================================================
# PURPOSE:
#   This file is the "brain" that coordinates everything.
#   It doesn't know HOW to attack a frame — that's the attack scripts' job.
#   It doesn't know HOW to read a video — that's video_utils' job.
#   Pipeline.py ONLY manages the flow:
#
#   For each video in input folder:
#     1. Check if already processed (resumability)
#     2. Extract frames + audio
#     3. Call the correct attack on all frames
#     4. Reconstruct video from attacked frames
#     5. Reattach original audio
#     6. Save final output
#     7. Clean up temp files
#     8. Update progress tracker
#
# RESUMABILITY — Why and How:
#   With 12,000 videos, processing will take hours/days.
#   If your computer crashes or you close the terminal, you DON'T want to
#   start over from video 1. The progress.json file acts like a bookmark.
#   Every time a video finishes, we mark it "done" in that file.
#   On restart, we skip everything marked "done".
#
# LIBRARIES:
#   json     — Built-in Python module for reading/writing progress.json
#   logging  — Built-in Python module for structured logging
#   tqdm     — Progress bar library (pip install tqdm)
#              Shows: [=====     ] 523/12000 videos | 4.2 it/s | ETA 44:32
#   pathlib  — Modern path handling
#   datetime — For logging timestamps
# =============================================================================

import os
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

# tqdm provides the progress bar
# Install: pip install tqdm
from tqdm import tqdm

# Import our utility and attack modules
# Using relative imports (. means "from the same package/folder")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import video_utils
from attacks import jpeg_attack, gaussian_attack, dct_attack, histogram_attack
from config import CONFIG


# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(log_file: str):
    """
    Configures Python's logging system to write to both:
    1. The terminal (so you can see progress while it runs)
    2. A log file (so you have a record after it finishes)

    Log format: 2025-06-01 14:32:11 | pipeline | INFO | Processing video 523
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),   # Write to file
            logging.StreamHandler()          # Write to terminal
        ]
    )

logger = logging.getLogger("pipeline")


# =============================================================================
# PROGRESS TRACKING
# =============================================================================
class ProgressTracker:
    """
    Manages the progress.json file for resumable processing.

    The JSON structure looks like:
    {
        "jpeg": {
            "video_001.mp4": "done",
            "video_002.mp4": "done",
            "video_003.mp4": "failed",
            "video_004.mp4": "pending"
        },
        "gaussian": { ... },
        ...
    }

    When the pipeline starts for an attack type, it reads this file,
    skips "done" videos, and retries "pending" and "failed" ones.
    """

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.data = self._load()

    def _load(self) -> dict:
        """Load existing progress or create empty structure."""
        if os.path.exists(self.progress_file):
            # Check if file has content before trying to load it
            if os.path.getsize(self.progress_file) > 0:
                with open(self.progress_file, "r") as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        # Return empty dict if the file is corrupted or not valid JSON
                        return {}
        return {}

    def _save(self):
        """Write current progress to JSON file."""
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_done(self, attack_type: str, video_name: str) -> bool:
        """Check if a video has already been processed for a given attack."""
        return self.data.get(attack_type, {}).get(video_name) == "done"

    def mark_done(self, attack_type: str, video_name: str):
        """Mark a video as successfully processed."""
        if attack_type not in self.data:
            self.data[attack_type] = {}
        self.data[attack_type][video_name] = "done"
        self._save()

    def mark_failed(self, attack_type: str, video_name: str):
        """Mark a video as failed (will be retried on next run)."""
        if attack_type not in self.data:
            self.data[attack_type] = {}
        self.data[attack_type][video_name] = "failed"
        self._save()

    def get_stats(self, attack_type: str) -> dict:
        """Return counts of done/failed/pending for a given attack type."""
        statuses = self.data.get(attack_type, {}).values()
        return {
            "done":    sum(1 for s in statuses if s == "done"),
            "failed":  sum(1 for s in statuses if s == "failed"),
            "pending": sum(1 for s in statuses if s == "pending"),
        }


# =============================================================================
# HELPER: GET VIDEO FILES FROM INPUT DIRECTORY
# =============================================================================
def get_video_files(input_dir: str, valid_extensions: list) -> list:
    """
    Recursively finds all video files in the input directory.

    WHY RECURSIVE?
    FakeAVCeleb is organized into subfolders by manipulation type:
      FakeVideo-RealAudio/
        subject_01/
          vid_001.mp4
          vid_002.mp4
        subject_02/
          ...

    We want to find ALL .mp4 files regardless of how deep they are nested.
    Path.rglob("*.mp4") does this recursively.

    Returns:
        Sorted list of Path objects for all video files found
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return []

    video_files = []
    for ext in valid_extensions:
        # rglob = recursive glob — finds files at any depth
        # "**/*" means "in any subfolder at any level"
        video_files.extend(input_path.rglob(f"*{ext}"))

    # Sort for consistent ordering across runs
    video_files = sorted(video_files)
    logger.info(f"Found {len(video_files)} video files in {input_dir}")

    return video_files


# =============================================================================
# CORE: PROCESS A SINGLE VIDEO WITH A GIVEN ATTACK
# =============================================================================
def process_single_video(
    video_path: Path,
    attack_type: str,
    output_dir: str,
) -> bool:
    """
    Processes one video through the complete attack pipeline:
    Extract → Attack → Reconstruct → Merge Audio → Save → Cleanup

    Args:
        video_path  : Path to the input video
        attack_type : One of "jpeg", "gaussian", "dct", "histogram"
        output_dir  : Folder where the attacked video will be saved

    Returns:
        True  if processing succeeded
        False if anything went wrong
    """

    video_name = video_path.name  # Just the filename, e.g. "video_001.mp4"

    # --- Create per-video temp folder ---
    # Each video gets its own temp folder: ./temp/video_001/
    # This prevents frame files from different videos mixing together
    video_stem = video_path.stem  # filename without extension
    temp_folder = os.path.join(CONFIG["paths"]["temp_dir"], video_stem)
    frames_folder = os.path.join(temp_folder, "frames")
    audio_path = os.path.join(temp_folder, "audio.wav")
    reconstructed_path = os.path.join(temp_folder, "reconstructed_no_audio.mp4")

    # Final output path mirrors the input subfolder structure
    # This preserves FakeAVCeleb's organization in the output
    relative_path = video_path.relative_to(CONFIG["paths"]["input_dir"])
    final_output_path = os.path.join(output_dir, str(relative_path))
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    try:
        # ── STEP 1: Extract frames ─────────────────────────────────────────
        os.makedirs(frames_folder, exist_ok=True)
        frame_paths = video_utils.extract_frames(
            str(video_path),
            frames_folder,
            fps=CONFIG["video"]["fps"]
        )

        if not frame_paths:
            logger.warning(f"No frames extracted from {video_name}")
            return False

        # ── STEP 2: Extract audio (saved separately, untouched) ────────────
        has_audio = video_utils.extract_audio(str(video_path), audio_path)

        # ── STEP 3: Apply the selected attack to all frames ────────────────
        # Each attack module has an attack_frames() function with the same
        # signature: attack_frames(frame_paths, **attack_params)
        # We select the right one based on attack_type string

        if attack_type == "jpeg":
            jpeg_attack.attack_frames(
                frame_paths,
                quality=CONFIG["jpeg"]["quality"]
            )

        elif attack_type == "gaussian":
            gaussian_attack.attack_frames(
                frame_paths,
                mean=CONFIG["gaussian"]["mean"],
                sigma=CONFIG["gaussian"]["sigma"]
            )

        elif attack_type == "dct":
            dct_attack.attack_frames(
                frame_paths,
                block_size=CONFIG["dct"]["block_size"],
                keep_fraction=CONFIG["dct"]["keep_fraction"]
            )

        elif attack_type == "histogram":
            histogram_attack.attack_frames(
                frame_paths,
                shift_range=CONFIG["histogram"]["shift_range"],
                clip_min=CONFIG["histogram"]["clip_min"],
                clip_max=CONFIG["histogram"]["clip_max"]
            )

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # ── STEP 4: Reconstruct video from attacked frames ─────────────────
        metadata = video_utils.get_video_metadata(str(video_path))
        fps = metadata.get("fps", CONFIG["video"]["fps"])

        video_utils.reconstruct_video(
            frame_paths,
            reconstructed_path,
            fps=fps,
            codec=CONFIG["video"]["codec"]
        )

        # ── STEP 5: Merge original audio back in ───────────────────────────
        if has_audio and os.path.exists(audio_path):
            video_utils.merge_audio_into_video(
                reconstructed_path,
                audio_path,
                final_output_path
            )
        else:
            # No audio in original — just copy the video-only version
            shutil.copy(reconstructed_path, final_output_path)

        # ── STEP 6: Cleanup temp folder ────────────────────────────────────
        video_utils.cleanup_temp(temp_folder)

        return True

    except Exception as e:
        logger.error(f"Failed processing {video_name}: {str(e)}", exc_info=True)
        # Clean up partial temp files on failure
        video_utils.cleanup_temp(temp_folder)
        return False


# =============================================================================
# MAIN PIPELINE: Run an entire attack type over all videos
# =============================================================================
def run_attack_pipeline(attack_type: str):
    """
    Runs the full statistical attack pipeline for one attack type.

    This is the function that run.py calls.
    It handles: finding files, skipping completed ones, showing progress,
    and logging results.

    Args:
        attack_type : One of "jpeg", "gaussian", "dct", "histogram"
    """

    # Validate attack type
    valid_attacks = ["jpeg", "gaussian", "dct", "histogram"]
    if attack_type not in valid_attacks:
        logger.error(f"Invalid attack type '{attack_type}'. Choose from: {valid_attacks}")
        return

    setup_logging(CONFIG["paths"]["log_file"])

    logger.info("=" * 60)
    logger.info(f"DeepTrust — Statistical Attack Pipeline")
    logger.info(f"Attack Type : {attack_type.upper()}")
    logger.info(f"Input Dir   : {CONFIG['paths']['input_dir']}")
    logger.info(f"Output Dir  : {CONFIG['paths']['output_subfolders'][attack_type]}")
    logger.info(f"Start Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # --- Load output directory for this attack ---
    output_dir = CONFIG["paths"]["output_subfolders"][attack_type]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CONFIG["paths"]["temp_dir"], exist_ok=True)

    # --- Load progress tracker ---
    tracker = ProgressTracker(CONFIG["paths"]["progress_file"])

    # --- Get all video files ---
    video_files = get_video_files(
        CONFIG["paths"]["input_dir"],
        CONFIG["pipeline"]["valid_extensions"]
    )

    if not video_files:
        logger.error("No video files found in input directory. "
                     "Update CONFIG['paths']['input_dir'] in config.py")
        return

    # --- Filter out already-completed videos ---
    if CONFIG["pipeline"]["skip_completed"]:
        pending = [v for v in video_files
                   if not tracker.is_done(attack_type, v.name)]
        skipped = len(video_files) - len(pending)
        if skipped > 0:
            logger.info(f"Resuming: skipping {skipped} already-completed videos")
    else:
        pending = video_files

    # --- Apply Dataset Splitting if specified ---
    if CONFIG["pipeline"].get("split_dataset", False):
        total_videos = len(pending)
        chunk_size = total_videos // 4
        
        # Determine which chunk belongs to which attack
        # By always slicing the exact same way, it's deterministic and safe
        if attack_type == "jpeg":
            pending = pending[0:chunk_size]
        elif attack_type == "gaussian":
            pending = pending[chunk_size:chunk_size*2]
        elif attack_type == "dct":
            pending = pending[chunk_size*2:chunk_size*3]
        elif attack_type == "histogram":
            pending = pending[chunk_size*3:] # Takes the remainder as well
            
        logger.info(f"Dataset Split enabled: processing chunk of {len(pending)} videos for {attack_type}")

    # --- Apply test limit if specified ---
    limit = CONFIG["pipeline"].get("limit")
    if limit is not None:
        pending = pending[:limit]
        logger.info(f"Limit applied: processing only first {len(pending)} videos")

    logger.info(f"Videos to process: {len(pending)}")

    # --- Process all pending videos with a progress bar ---
    success_count = 0
    fail_count = 0

    # tqdm wraps the iterable and displays a progress bar
    # desc=  → label shown on the left
    # unit=  → what each iteration represents
    for video_path in tqdm(pending, desc=f"{attack_type} attack", unit="video"):

        success = process_single_video(
            video_path,
            attack_type,
            output_dir
        )

        if success:
            tracker.mark_done(attack_type, video_path.name)
            success_count += 1
        else:
            tracker.mark_failed(attack_type, video_path.name)
            fail_count += 1

    # --- Final summary ---
    logger.info("=" * 60)
    logger.info(f"Pipeline complete for: {attack_type.upper()}")
    logger.info(f"  Successfully processed : {success_count}")
    logger.info(f"  Failed                 : {fail_count}")
    logger.info(f"  End Time               : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
