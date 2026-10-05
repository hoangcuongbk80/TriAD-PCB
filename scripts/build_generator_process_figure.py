from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    title: str,
    lines: list[str],
) -> None:
    draw.rounded_rectangle(box, radius=15, fill=fill, outline="#475569", width=2)
    x1, y1, x2, _ = box
    title_font, body_font = font(20, bold=True), font(16)
    title_width = draw.textlength(title, font=title_font)
    draw.text(((x1 + x2 - title_width) / 2, y1 + 15), title, fill="#111827", font=title_font)
    for index, line in enumerate(lines):
        line_width = draw.textlength(line, font=body_font)
        draw.text(
            ((x1 + x2 - line_width) / 2, y1 + 53 + 25 * index),
            line,
            fill="#334155",
            font=body_font,
        )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#111827",
) -> None:
    draw.line([start, end], fill=color, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 13
    for offset in (0.55, -0.55):
        point = (
            end[0] - size * math.cos(angle + offset),
            end[1] - size * math.sin(angle + offset),
        )
        draw.line([end, point], fill=color, width=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the prompt-conditioned generation diagram")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "latex" / "generator_process.png"
    )
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        raise ValueError("The output must use the .png extension")
    width, height = 1600, 385
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    heading = "Training-only prompt-conditioned feature generation"
    heading_font = font(27, bold=True)
    heading_width = draw.textlength(heading, font=heading_font)
    draw.text(((width - heading_width) / 2, 14), heading, fill="#111827", font=heading_font)

    rounded_box(draw, (25, 95, 250, 205), "#dbeafe", "Real patch feature", ["z from a normal image", "L2-normalized embedding"])
    rounded_box(draw, (25, 230, 250, 340), "#ede9fe", "Prompt condition", ["normal prompt n", "or abnormal prompt a"])
    rounded_box(draw, (340, 145, 565, 275), "#f1f5f9", "Concatenation", ["[z ; n] or [z ; a]", "shared input dimension"])
    rounded_box(draw, (655, 145, 880, 275), "#fee2e2", "Generator G", ["two-layer MLP", "512-D output"])
    rounded_box(draw, (970, 78, 1225, 188), "#dcfce7", "Pseudo-normal", ["aligned with n", "close to normal manifold"])
    rounded_box(draw, (970, 230, 1225, 340), "#ffedd5", "Pseudo-abnormal", ["aligned with a", "separated from n"])
    rounded_box(draw, (1320, 78, 1565, 188), "#ecfccb", "Positive supervision", ["normal alignment", "in L_abn"])
    rounded_box(draw, (1320, 230, 1565, 340), "#fef2f2", "Hard negative", ["L_nce", "and L_abn"])

    arrow(draw, (250, 150), (340, 190))
    arrow(draw, (250, 285), (340, 230))
    arrow(draw, (565, 210), (655, 210))
    arrow(draw, (880, 190), (970, 133), "#15803d")
    arrow(draw, (880, 230), (970, 285), "#c2410c")
    arrow(draw, (1225, 133), (1320, 133), "#15803d")
    arrow(draw, (1225, 285), (1320, 285), "#c2410c")

    note_font = font(15)
    draw.text((341, 292), "The same generator is used for both conditions.", fill="#475569", font=note_font)
    draw.text((25, 355), "The generator is removed after training and adds no inference-time latency.", fill="#7f1d1d", font=font(16, bold=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
