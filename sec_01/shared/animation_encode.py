"""Encode frame sequences into GIF and optional MP4 outputs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image


def encode_gif_from_frames(frames: list[Path], output_path: Path, fps: int) -> None:
    """Encode GIF from frame image sequence.

    Args:
        frames: Sorted frame file paths.
        output_path: Destination GIF path.
        fps: Frames per second.
    """

    if not frames:
        raise ValueError("No frames were provided for GIF encoding.")

    duration_ms = int(1000 / max(fps, 1))
    images = [Image.open(frame) for frame in frames]
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
    finally:
        for image in images:
            image.close()


def encode_mp4_with_ffmpeg(
    frames_pattern: str,
    output_path: Path,
    fps: int,
    frame_count: int | None = None,
) -> bool:
    """Encode MP4 by calling ffmpeg.

    Args:
        frames_pattern: Input pattern compatible with ffmpeg, e.g. B02_%04d.png.
        output_path: Destination MP4 path.
        fps: Frames per second.
        frame_count: Optional number of frames to encode from the sequence start.

    Returns:
        True when MP4 was generated.
    """

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False

    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(max(fps, 1)),
        "-i",
        frames_pattern,
    ]
    if frame_count is not None:
        cmd.extend(["-frames:v", str(max(frame_count, 1))])
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return completed.returncode == 0
