"""Browser-compatible video encoding helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class H264VideoWriter:
    """Write MP4 with H.264 (yuv420p + faststart) for web/desktop players."""

    def __init__(self, path: str | Path, fps: float, frame_size: tuple[int, int]):
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg not found; install ffmpeg for H.264 export")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        width, height = frame_size
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                "bgr24",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin closed")
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to encode {self.path}")


def transcode_to_h264(src: Path, dst: Path | None = None) -> Path:
    """Re-encode an existing video to H.264 for compatibility."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found")

    src = Path(src)
    if dst is None:
        dst = src.with_name(f"{src.stem}_h264{src.suffix}")
    else:
        dst = Path(dst)

    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp.replace(dst)
    return dst


def create_video_writer(path: str | Path, fps: float, frame_size: tuple[int, int]):
    """Prefer H.264 via ffmpeg; fall back to OpenCV mp4v + transcode."""
    try:
        return H264VideoWriter(path, fps, frame_size), "h264"
    except RuntimeError:
        import cv2

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        return writer, "mp4v"
