#!/usr/bin/env python3
"""Draws the WhisperScribe icon and writes assets/icon.ico (+ icon.png preview).

    pip install pillow numpy
    python assets/make_icon.py

The .ico is already committed, so you only need this to change the design.
Small sizes (<= 32 px) use a simplified glyph so the icon stays readable in the taskbar.
"""
import io
import os
import struct

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]

# size -> (bar width px, gap px, five bar heights px) for the pixel-snapped small glyphs
SMALL = {
    16: (2, 1, [6, 10, 14, 10, 6]),
    20: (2, 2, [8, 12, 16, 12, 8]),
    24: (2, 2, [8, 14, 18, 14, 8]),
    32: (4, 2, [10, 18, 24, 18, 10]),
}

TOP = (99, 102, 241)      # indigo
BOTTOM = (14, 165, 233)   # sky blue
WHITE = (255, 255, 255)


def gradient(s):
    """Vertical gradient TOP -> BOTTOM as an RGBA image of size s x s."""
    t = np.linspace(0.0, 1.0, s)[:, None]
    top = np.array(TOP, dtype=np.float32)
    bot = np.array(BOTTOM, dtype=np.float32)
    rgb = top + (bot - top) * t[:, :, None]            # (s, 1, 3)
    rgb = np.repeat(rgb, s, axis=1)                    # (s, s, 3)
    a = np.full((s, s, 1), 255, dtype=np.float32)
    return Image.fromarray(np.concatenate([rgb, a], axis=2).astype(np.uint8), "RGBA")


def rounded_bar(draw, cx, cy, w, h, fill):
    draw.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=w / 2, fill=fill)


def render(size):
    ss = 8 if size <= 48 else 4                        # supersampling factor
    s = size * ss
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # background tile with rounded corners
    mask = Image.new("L", (s, s), 0)
    pad = 0.03 * s
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, s - pad, s - pad], radius=0.23 * s, fill=255)
    img.paste(gradient(s), (0, 0), mask)

    d = ImageDraw.Draw(img)
    if size <= 32:
        # simplified glyph: 5 bars whose edges sit exactly on whole pixels, so they stay crisp
        bw, gap, heights = SMALL[size]
        total = 5 * bw + 4 * gap
        x0 = (size - total) // 2
        for i, h in enumerate(heights):
            left = (x0 + i * (bw + gap)) * ss
            right = left + bw * ss
            top = (size // 2 - h // 2) * ss
            bottom = top + h * ss
            d.rounded_rectangle([left, top, right, bottom], radius=bw * ss / 2, fill=WHITE)
    else:
        # full glyph: waveform with two transcript lines underneath
        heights = [0.16, 0.30, 0.48, 0.62, 0.48, 0.30, 0.16]
        n = len(heights)
        bw = 0.072 * s
        gap = 0.048 * s
        total = n * bw + (n - 1) * gap
        x0 = (s - total) / 2 + bw / 2
        cy = 0.40 * s
        for i, h in enumerate(heights):
            rounded_bar(d, x0 + i * (bw + gap), cy, bw, h * s, WHITE)

        lx = x0 - bw / 2                                # left edge shared with the waveform
        lh = 0.058 * s
        for width, y, alpha in [(total, 0.755 * s, 235), (total * 0.62, 0.855 * s, 190)]:
            d.rounded_rectangle([lx, y - lh / 2, lx + width, y + lh / 2], radius=lh / 2,
                                fill=WHITE + (alpha,))

    return img.resize((size, size), Image.LANCZOS)


# ---- minimal ICO writer: classic 32-bit DIB for small sizes, PNG for >= 64 px ----------
def dib_bytes(img):
    w, h = img.size
    px = np.array(img.convert("RGBA"), dtype=np.uint8)
    bgra = px[:, :, [2, 1, 0, 3]][::-1]                 # BGRA, bottom-up rows
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, w * h * 4, 0, 0, 0, 0)
    and_row = ((w + 31) // 32) * 4
    return header + bgra.tobytes() + bytes(and_row * h)  # all-zero AND mask (alpha does the work)


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, format="PNG", optimize=True)
    return b.getvalue()


def write_ico(path, images):
    blobs = [png_bytes(im) if im.size[0] >= 64 else dib_bytes(im) for im in images]
    out = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    for im, blob in zip(images, blobs):
        w, h = im.size
        out += struct.pack("<BBBBHHII", w % 256, h % 256, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    with open(path, "wb") as f:
        f.write(out + b"".join(blobs))


if __name__ == "__main__":
    images = [render(sz) for sz in SIZES]
    write_ico(os.path.join(HERE, "icon.ico"), images)
    render(512).save(os.path.join(HERE, "icon.png"))
    print("wrote icon.ico (%s) and icon.png" % ", ".join(str(s) for s in SIZES))
