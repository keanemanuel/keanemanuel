#!/usr/bin/env python3
"""
dotify.py — turn a photo into a dot-matrix SVG portrait for a GitHub README.

Usage:
    python dotify.py input.png -o assets/portrait \
        --cols 100 --detail 0.5 --color --animate
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def build_svg(img: Image.Image, cols: int, detail: float, color: bool,
              bg: str, max_dot: float, min_dot: float,
              animate: bool = False, anim_duration: float = 2.5,
              alpha: np.ndarray = None, alpha_threshold: float = 0.7) -> str:
    w, h = img.size
    cell = w / cols
    rows = max(1, round(h / cell))
    cell_h = h / rows

    blur_radius = max(1.0, (w / cols) * 0.6)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    small = blurred.resize((cols, rows), Image.LANCZOS)
    rgb = np.asarray(small.convert("RGB"), dtype=np.float32)
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0

    gray = 0.5 + (gray - 0.5) * (0.4 + 1.6 * detail)
    gray = np.clip(gray, 0, 1)

    # Resize the alpha mask the same way, nearest-ish via LANCZOS on a
    # single-channel image, then threshold: >0.5 means "real content here"
    alpha_small = None
    if alpha is not None:
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
        alpha_img = alpha_img.resize((cols, rows), Image.NEAREST)
        alpha_small = np.asarray(alpha_img, dtype=np.float32) / 255.0

    svg_w, svg_h = w, h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" '
        f'width="{svg_w}" height="{svg_h}">',
        f'<rect x="0" y="0" width="{svg_w}" height="{svg_h}" fill="{bg}"/>',
    ]

    if animate:
        parts.append(f"""<defs>
  <clipPath id="reveal">
    <rect x="0" y="0" width="{svg_w}" height="0">
      <animate attributeName="height" from="0" to="{svg_h}"
               dur="{anim_duration}s" begin="0s" fill="freeze"
               calcMode="spline" keySplines="0.25 0.1 0.25 1" />
    </rect>
  </clipPath>
</defs>""")
        parts.append('<g clip-path="url(#reveal)">')

    for ry in range(rows):
        for rx in range(cols):
            # If we have real alpha data, that alone decides whether this
            # cell has content — a white sleeve (alpha=1) always draws,
            # a transparent background (alpha=0) never does.
            if alpha_small is not None:
                if alpha_small[ry, rx] < 0.9999:
                    continue
                intensity = max(0.15, 1.0 - gray[ry, rx])
            else:
                intensity = 1.0 - gray[ry, rx]
                if intensity <= 0.08:
                    continue

            radius = (min_dot + (max_dot - min_dot) * intensity) * (cell / 2)
            cx = (rx + 0.5) * cell
            cy = (ry + 0.5) * cell_h

            if color:
                r, g, b = rgb[ry, rx]
                fill = f"rgb({int(r)},{int(g)},{int(b)})"
            else:
                fill = "#e8823c"

            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.2f}" fill="{fill}"/>'
            )

    if animate:
        parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Convert a photo to a dot-matrix SVG portrait.")
    ap.add_argument("input", type=Path, help="input image path (png/jpg)")
    ap.add_argument("-o", "--out", type=Path, required=True,
                     help="output path WITHOUT extension, e.g. assets/portrait")
    ap.add_argument("--cols", type=int, default=100, help="number of dot columns")
    ap.add_argument("--equalize", action="store_true",
                     help="apply histogram equalization for stronger contrast")
    ap.add_argument("--detail", type=float, default=0.5,
                     help="0-1, how much brightness contrast affects dot size")
    ap.add_argument("--color", action="store_true",
                     help="sample colour from the source image per dot")
    ap.add_argument("--bg", default="#0d1117", help="background fill colour")
    ap.add_argument("--max-dot", type=float, default=0.95, help="max dot radius as fraction of cell/2")
    ap.add_argument("--min-dot", type=float, default=0.05, help="min dot radius as fraction of cell/2")
    ap.add_argument("--animate", action="store_true",
                     help="reveal the portrait top-to-bottom on load (plays once)")
    ap.add_argument("--anim-duration", type=float, default=2.5,
                     help="seconds for the top-to-bottom reveal animation")
    ap.add_argument("--alpha-threshold", type=float, default=0.7,
                     help="0-1, how opaque a pixel must be to draw a dot")
    ap.add_argument("--saturation", type=float, default=1.6,
                     help="colour saturation multiplier, 1.0 = unchanged")
    args = ap.parse_args()

    raw = Image.open(args.input)
    alpha = None
    if raw.mode in ("RGBA", "LA") or (raw.mode == "P" and "transparency" in raw.info):
        raw = raw.convert("RGBA")
        alpha = np.asarray(raw.split()[-1], dtype=np.float32) / 255.0
        # Composite onto white just so colour sampling has something sane
        # to read for fully-opaque and semi-transparent pixels alike.
        canvas = Image.new("RGB", raw.size, (8, 6, 5))
        canvas.paste(raw, mask=raw.split()[-1])
        img = canvas
    else:
        img = raw.convert("RGB")

    img = ImageEnhance.Color(img).enhance(args.saturation)

    if args.equalize:
        gray = ImageOps.equalize(img.convert("L"))
        if args.color:
            hsv = img.convert("HSV")
            h, s, _ = hsv.split()
            img = Image.merge("HSV", (h, s, gray)).convert("RGB")
        else:
            img = gray.convert("RGB")

    svg = build_svg(
        img,
        cols=args.cols,
        detail=args.detail,
        color=args.color,
        bg=args.bg,
        max_dot=args.max_dot,
        min_dot=args.min_dot,
        animate=args.animate,
        anim_duration=args.anim_duration,
        alpha=alpha,
        alpha_threshold=args.alpha_threshold,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_path = args.out.with_suffix(".svg")
    out_path.write_text(svg)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
