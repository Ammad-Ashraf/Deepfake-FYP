# =============================================================================
# video_utils.py — Video I/O Utilities for the Statistical Attack Pipeline
# =============================================================================
# PURPOSE:
#   This file handles ALL the "plumbing" of working with video files.
#   No attack logic lives here — only the mechanical work of:
#     1. Splitting a video into individual frames (images)
#     2. Extracting the audio track separately
#     3. Rebuilding a video from modified frames
#     4. Reattaching the original audio to the new video
#
# WHY SEPARATE THIS?
#   Each attack script should only worry about "how do I transform this frame?"
#   NOT about "how do I open an mp4 file?". Keeping I/O separate means:
#     - If OpenCV changes its API, you fix it in ONE place
#     - Attack scripts stay clean and easy to read
#     - You can test attacks on single images without needing a video
#
# LIBRARIES USED:
#   cv2 (OpenCV) — Industry-standard computer vision library
#                  Used for: reading video files, extracting frames,
#                  writing video files frame-by-frame
#                  Install: pip install opencv-python
#
#   subprocess   — Python's built-in module for running shell commands
#                  Used to call FFmpeg from inside Python
#                  FFmpeg handles audio extraction/merging far better than OpenCV
#                  (OpenCV doesn't handle audio at all)
#
#   os, pathlib  — Built-in Python modules for file/folder operations
#
#   numpy        — Numerical array library (frames are stored as numpy arrays)
#                  Install: pip install numpy
# =============================================================================

import cv2          # OpenCV: video reading and frame writing
import os           # File path operations
import subprocess   # For calling FFmpeg shell commands
import shutil       # For deleting temp folders
import numpy as np  # Frame data is stored as numpy arrays
import logging      # For writing structured logs

from pathlib import Path  # Modern, clean path handling (better than os.path)


# =============================================================================
# LOGGING SETUP
# =============================================================================
# We create a logger specifically for this module.
# When other files import video_utils, their log messages will say "video_utils"
# making it easy to trace where each message came from.
logger = logging.getLogger("video_utils")


# =============================================================================
# FUNCTION: extract_frames
# =============================================================================
def extract_frames(video_path: str, output_folder: str, fps: int = 30) -> list:
    """
    Reads an .mp4 video file and saves every frame as a .png image.

    WHY PNG? Because PNG is lossless — saving frames as JPEG would introduce
    compression artifacts BEFORE we even apply our attack. We want clean frames
    going into each attack function.

    Args:
        video_path    : Full path to the input .mp4 file
        output_folder : Folder where frame images will be saved
        fps           : Expected frames per second (used for logging only)

    Returns:
        List of file paths to the extracted frame images, in order.
        Example: ["./temp/vid001/frame_00001.png", "frame_00002.png", ...]

    Raises:
        FileNotFoundError : If the video file doesn't exist
        RuntimeError      : If OpenCV can't open the video
    """

    # --- Validate input ---
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # --- Create output folder if it doesn't exist ---
    os.makedirs(output_folder, exist_ok=True)

    # --- Open the video file using OpenCV ---
    # cv2.VideoCapture is OpenCV's class for reading video files
    # It supports mp4, avi, mkv, and most common formats
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    # --- Read video metadata ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps   = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.debug(f"Video: {Path(video_path).name} | "
                 f"{total_frames} frames | {actual_fps}fps | {width}x{height}")

    # --- Extract frames one by one ---
    frame_paths = []
    frame_index = 0

    while True:
        # cap.read() returns (success_bool, frame_as_numpy_array)
        # When the video ends, ret becomes False
        ret, frame = cap.read()

        if not ret:
            # End of video reached — stop reading
            break

        # Zero-pad the frame number so files sort correctly alphabetically
        # frame_00001.png, frame_00002.png ... frame_01000.png
        frame_filename = f"frame_{frame_index:05d}.png"
        frame_path = os.path.join(output_folder, frame_filename)

        # cv2.imwrite saves the numpy array as an image file
        # PNG format preserves exact pixel values (lossless)
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_index += 1

    # --- Always release the video capture object ---
    # This frees the file handle — important on Windows where open files are locked
    cap.release()

    logger.debug(f"Extracted {frame_index} frames to {output_folder}")
    return frame_paths


# =============================================================================
# FUNCTION: extract_audio
# =============================================================================
def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """
    Extracts the audio track from a video file and saves it as a .wav file.

    WHY FFMPEG INSTEAD OF OPENCV?
    OpenCV is a computer vision library — it reads VIDEO (images) but completely
    ignores audio. FFmpeg is a multimedia powerhouse that handles both.
    We call FFmpeg as a subprocess (shell command) from inside Python.

    WHY SAVE AUDIO SEPARATELY?
    Our statistical attacks only modify visual frames. The audio stays completely
    untouched. After attacking the frames, we reconstruct the video and then
    REATTACH this original audio. This is critical because:
    1. Audio-visual sync is preserved (important for Iteration 2's multimodal model)
    2. Ammad's work doesn't overlap with audio domain (that's Iteration 2's job)

    Args:
        video_path         : Full path to the input .mp4 file
        output_audio_path  : Where to save the extracted .wav file

    Returns:
        True  if audio was extracted successfully
        False if the video has no audio track (some videos don't)
    """

    # Build the FFmpeg command as a list of arguments
    # subprocess.run() takes either a string or a list — list is safer (no shell injection)
    #
    # Breakdown of FFmpeg flags:
    #   -i video_path    → input file
    #   -vn              → "video no" — ignore the video stream, only process audio
    #   -acodec pcm_s16le → encode audio as uncompressed 16-bit PCM (standard WAV)
    #   -y               → overwrite output file if it already exists
    #   output_audio_path → where to save the .wav file
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-y",
        output_audio_path
    ]

    try:
        # Run the FFmpeg command
        # capture_output=True hides FFmpeg's verbose output from your terminal
        # check=True raises an exception if FFmpeg returns an error code
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            check=True
        )
        logger.debug(f"Audio extracted: {output_audio_path}")
        return True

    except subprocess.CalledProcessError:
        # FFmpeg returned non-zero exit code — likely no audio stream in this video
        logger.warning(f"No audio track found in: {video_path}")
        return False

    except FileNotFoundError:
        # FFmpeg is not installed or not in system PATH
        logger.error("FFmpeg not found. Install FFmpeg and add it to your PATH.")
        raise


# =============================================================================
# FUNCTION: reconstruct_video
# =============================================================================
def reconstruct_video(
    frame_paths: list,
    output_video_path: str,
    fps: float = 30.0,
    codec: str = "mp4v"
) -> str:
    """
    Takes a list of modified frame images and assembles them back into a video.

    This is the REVERSE of extract_frames(). After each attack function has
    transformed the frames, this function stitches them back into an .mp4 file.
    At this stage, the output video has NO AUDIO — audio is added in the next step.

    Args:
        frame_paths       : Ordered list of paths to modified frame .png files
        output_video_path : Where to save the reconstructed video (no audio yet)
        fps               : Frame rate for the output video
        codec             : Video codec to use ("mp4v" for .mp4 files)

    Returns:
        Path to the reconstructed video file (still audio-less)

    Note:
        The frame_paths list MUST be in correct order (frame_00001 before frame_00002)
        We sort them to be safe — alphabetical sort works because of zero-padding
    """

    if not frame_paths:
        raise ValueError("frame_paths is empty — no frames to reconstruct video from")

    # Sort frames to ensure correct temporal order
    # Zero-padded names like frame_00001.png sort correctly alphabetically
    frame_paths = sorted(frame_paths)

    # --- Read the first frame to get video dimensions ---
    # We need width and height to initialize the VideoWriter
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    # frame.shape returns (height, width, channels)
    # [:2] takes just height and width, ignoring the color channel count

    # --- Initialize OpenCV VideoWriter ---
    # VideoWriter is OpenCV's class for creating video files frame by frame
    # Arguments: output path, codec (4-char code), fps, (width, height)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # *codec unpacks "mp4v" into 'm','p','4','v' — VideoWriter_fourcc needs 4 chars
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_video_path}")

    # --- Write frames one by one ---
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Skipping unreadable frame: {frame_path}")
            continue
        out.write(frame)

    # Release the VideoWriter (flushes buffer and finalizes the file)
    out.release()

    logger.debug(f"Video reconstructed: {output_video_path} ({len(frame_paths)} frames)")
    return output_video_path


# =============================================================================
# FUNCTION: merge_audio_into_video
# =============================================================================
def merge_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str
) -> str:
    """
    Merges a separate audio file back into a video file.

    This is the final step after reconstruction. The attacked video has
    correct modified visuals but no sound. This function reattaches the
    original audio extracted at the beginning.

    WHY FFMPEG AGAIN?
    OpenCV's VideoWriter cannot write audio. FFmpeg handles this perfectly
    with a simple muxing (mixing) command.

    Args:
        video_path  : Path to the video file WITHOUT audio
        audio_path  : Path to the original .wav audio file
        output_path : Where to save the final video WITH audio

    Returns:
        Path to the final output video file
    """

    # FFmpeg command to combine video + audio streams:
    # -i video_path     → first input: video stream
    # -i audio_path     → second input: audio stream
    # -c:v copy         → "codec video copy" — don't re-encode video, just copy it
    # -c:a aac          → encode audio as AAC (standard for .mp4 containers)
    # -shortest         → end output when the shorter stream ends
    #                     (prevents silent padding if audio is longer than video)
    # -y                → overwrite output if it exists
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-y",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        logger.debug(f"Audio merged into: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        # If merging fails, return the video-only version rather than failing entirely
        logger.warning(f"Audio merge failed for {video_path}. Saving video-only version.")
        shutil.copy(video_path, output_path)
        return output_path


# =============================================================================
# FUNCTION: get_video_metadata
# =============================================================================
def get_video_metadata(video_path: str) -> dict:
    """
    Returns basic metadata about a video file.
    Used by pipeline.py for logging and validation.

    Returns a dictionary with: fps, width, height, total_frames, duration_seconds
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {}

    fps           = cap.get(cv2.CAP_PROP_FPS)
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_secs = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps":              fps,
        "width":            width,
        "height":           height,
        "total_frames":     total_frames,
        "duration_seconds": round(duration_secs, 2),
    }


# =============================================================================
# FUNCTION: cleanup_temp
# =============================================================================
def cleanup_temp(temp_folder: str):
    """
    Deletes a temporary folder and all its contents.

    Called after each video is fully processed to free disk space.
    With 12,000 videos, not cleaning up would fill your hard drive quickly.

    Args:
        temp_folder : Path to the folder to delete
    """
    if os.path.exists(temp_folder):
        # ignore_errors=True prevents Windows PermissionError ([WinError 32]) 
        # from crashing the entire pipeline if a file is temporarily locked
        shutil.rmtree(temp_folder, ignore_errors=True)
        logger.debug(f"Cleaned up temp folder: {temp_folder}")
