"""Generates res/crosshair.ico (active) and res/crosshair_off.ico (inactive).

Usage: python tools/make_icons.py      (requires Pillow)
"""

import os

from PIL import Image, ImageDraw

SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "res")


def draw_icon(size, color, outline):
    # Draw at 8x and downsample for smooth edges.
    scale = 8
    n = size * scale
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = n / 2
    # Line widths in final pixels, kept at least ~1.5 px so small icons stay legible.
    line = max(1.6, size / 11) * scale
    ow = max(1.0, size / 22) * scale
    ring_r = n * 0.33
    gap = n * 0.10
    reach = n * 0.47   # arm end, measured from the center

    def ring(width, fill):
        d.ellipse([c - ring_r - width / 2, c - ring_r - width / 2,
                   c + ring_r + width / 2, c + ring_r + width / 2], outline=fill, width=int(width))

    def arms(width, extra, fill):
        h = width / 2
        for (x0, y0, x1, y1) in [
            (c + gap - extra, c - h, c + reach + extra, c + h),   # right
            (c - reach - extra, c - h, c - gap + extra, c + h),   # left
            (c - h, c + gap - extra, c + h, c + reach + extra),   # bottom
            (c - h, c - reach - extra, c + h, c - gap + extra),   # top
        ]:
            d.rectangle([x0, y0, x1, y1], fill=fill)

    def dot(r, fill):
        d.ellipse([c - r, c - r, c + r, c + r], fill=fill)

    # Small icons: a plain cross (a ring would turn into mush at 16 px).
    small = size <= 24
    if small:
        line = 2.0 * scale
        ow = 1.0 * scale
        gap = n * 0.12
        reach = n * 0.5 - ow

    # Outline pass, then color pass.
    if not small:
        ring(line + 2 * ow, outline)
    arms(line + 2 * ow, ow, outline)
    if not small:
        dot(line * 0.75 + ow, outline)
        ring(line, color)
    arms(line, 0, color)
    if not small:
        dot(line * 0.75, color)

    return img.resize((size, size), Image.LANCZOS)


def save_ico(path, color, outline):
    images = [draw_icon(s, color, outline) for s in SIZES]
    images[-1].save(path, format="ICO", sizes=[(s, s) for s in SIZES],
                    append_images=images[:-1])


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    save_ico(os.path.join(OUT_DIR, "crosshair.ico"), (40, 235, 90, 255), (10, 20, 12, 255))
    save_ico(os.path.join(OUT_DIR, "crosshair_off.ico"), (150, 150, 150, 255), (40, 40, 40, 255))
    print("icons written to", os.path.normpath(OUT_DIR))
