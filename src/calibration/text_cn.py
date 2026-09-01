"""Draw Chinese text on OpenCV BGR images (Hershey fonts cannot render CJK)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
]


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size, index=0)
        except OSError:
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


@lru_cache(maxsize=256)
def _render_chip(
    text: str,
    font_size: int,
    fill_rgb: tuple[int, int, int],
    stroke_rgb: tuple[int, int, int],
    stroke_width: int,
) -> np.ndarray:
    """Return BGRA chip (H, W, 4) for a short label — cached."""
    font = _load_font(font_size)
    # Measure
    dummy = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(dummy)
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = font_size * max(len(text), 1), font_size + 4
    pad = stroke_width + 2
    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (pad, pad), text, font=font, fill=(*fill_rgb, 255),
        stroke_width=stroke_width, stroke_fill=(*stroke_rgb, 255),
    )
    arr = np.asarray(img)
    # RGBA → BGRA
    return arr[:, :, [2, 1, 0, 3]].copy()


def paste_chip(
    img_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_size: int = 18,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    stroke_bgr: tuple[int, int, int] = (0, 0, 0),
    stroke_width: int = 2,
) -> None:
    """Paste a cached text chip; org is baseline-ish (x, y)."""
    if not text:
        return
    fill_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    stroke_rgb = (int(stroke_bgr[2]), int(stroke_bgr[1]), int(stroke_bgr[0]))
    chip = _render_chip(text, int(font_size), fill_rgb, stroke_rgb, int(stroke_width))
    h, w = chip.shape[:2]
    x, y = int(org[0]), int(org[1]) - h  # baseline → top
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return
    cx1, cy1 = x1 - x, y1 - y
    cx2, cy2 = cx1 + (x2 - x1), cy1 + (y2 - y1)
    region = chip[cy1:cy2, cx1:cx2]
    alpha = region[:, :, 3:4].astype(np.float32) / 255.0
    bgr = region[:, :, :3].astype(np.float32)
    dst = img_bgr[y1:y2, x1:x2].astype(np.float32)
    img_bgr[y1:y2, x1:x2] = (bgr * alpha + dst * (1.0 - alpha)).astype(np.uint8)


def draw_hud_banner(
    width: int,
    lines: Sequence[str],
    *,
    font_size: int = 22,
    line_h: int = 30,
    bg: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    """Render a slim top HUD (BGR) with Chinese — much cheaper than full-frame PIL."""
    height = max(line_h * len(lines) + 12, line_h + 12)
    banner = np.full((height, width, 3), bg, dtype=np.uint8)
    y = 28
    for line in lines:
        paste_chip(
            banner, line, (12, y),
            font_size=font_size, color_bgr=(255, 255, 255),
            stroke_bgr=(0, 0, 0), stroke_width=2,
        )
        y += line_h
    return banner


def draw_texts_cn(
    img_bgr: np.ndarray,
    items: Sequence[tuple[str, tuple[int, int], int, tuple[int, int, int]]],
    *,
    stroke_bgr: tuple[int, int, int] | None = (0, 0, 0),
    stroke_width: int = 2,
) -> np.ndarray:
    stroke = stroke_bgr or (0, 0, 0)
    for text, org, font_size, color_bgr in items:
        paste_chip(
            img_bgr, text, org,
            font_size=font_size, color_bgr=color_bgr,
            stroke_bgr=stroke, stroke_width=stroke_width,
        )
    return img_bgr


def put_text_cn(
    img_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_size: int = 22,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    stroke_bgr: tuple[int, int, int] | None = (0, 0, 0),
    stroke_width: int = 2,
) -> np.ndarray:
    paste_chip(
        img_bgr, text, org,
        font_size=font_size, color_bgr=color_bgr,
        stroke_bgr=stroke_bgr or (0, 0, 0),
        stroke_width=stroke_width,
    )
    return img_bgr
