from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 1080, 420


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis file was not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Analysis file is empty: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], columns: set[str], path: Path) -> None:
    missing = columns.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


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


def centred(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, text_font) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, fill="#303030", font=text_font)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str,
    width: int,
    dash: tuple[int, int] | None,
) -> None:
    if dash is None:
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    dash_length, gap = dash
    for start, end in zip(points, points[1:], strict=False):
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = max((dx * dx + dy * dy) ** 0.5, 1e-8)
        position = 0.0
        while position < distance:
            stop = min(position + dash_length, distance)
            first = (start[0] + dx * position / distance, start[1] + dy * position / distance)
            second = (start[0] + dx * stop / distance, start[1] + dy * stop / distance)
            draw.line([first, second], fill=fill, width=width)
            position += dash_length + gap


def line_plot(image: Image.Image, rows: list[dict[str, str]]) -> None:
    draw = ImageDraw.Draw(image)
    x0, width = 0, WIDTH // 2
    left, right, bottom, top = x0 + 76, x0 + width - 24, HEIGHT - 60, 50
    max_step = max(max(float(row["stream_step"]) for row in rows), 1.0)
    small, title = font(14), font(16, bold=True)
    draw.line((left, bottom, left, top), fill="#333333", width=2)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    for value in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = bottom - value / 0.5 * (bottom - top)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((left - 43, y - 8), f"{value:.1f}", fill="#303030", font=small)
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = left + fraction * (right - left)
        centred(draw, (x, bottom + 10), f"{fraction * max_step:.0f}", small)
    centred(draw, ((left + right) / 2, HEIGHT - 25), "Streaming step", small)
    draw.text((7, (top + bottom) / 2 - 8), "Fusion weight", fill="#303030", font=small)

    series = [
        ("lambda_text", "std_lambda_text", "Text", "#295cba", None),
        ("lambda_history", "std_lambda_history", "Online", "#d13d33", (7, 4)),
        ("lambda_reference", "std_lambda_reference", "Reference", "#299e5c", (2, 4)),
    ]
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for key, std_key, _, color, _ in series:
        upper, lower = [], []
        for row in rows:
            x = left + float(row["stream_step"]) / max_step * (right - left)
            mean, deviation = float(row[key]), float(row[std_key])
            upper.append((x, bottom - min(0.5, mean + deviation) / 0.5 * (bottom - top)))
            lower.append((x, bottom - max(0.0, mean - deviation) / 0.5 * (bottom - top)))
        rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        overlay_draw.polygon(upper + list(reversed(lower)), fill=(*rgb, 28))
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    for key, _, _, color, dash in series:
        points = [
            (
                left + float(row["stream_step"]) / max_step * (right - left),
                bottom - float(row[key]) / 0.5 * (bottom - top),
            )
            for row in rows
        ]
        dashed_line(draw, points, color, 3, dash)
    legend_x = left
    for _, _, label, color, dash in series:
        dashed_line(draw, [(legend_x, 32), (legend_x + 30, 32)], color, 3, dash)
        draw.text((legend_x + 38, 23), label, fill="#303030", font=small)
        legend_x += 38 + int(draw.textlength(label, font=small)) + 28
    draw.text((7, 5), "(a) Confidence-aware fusion during warm-up", fill="#202020", font=title)


def projection_plot(image: Image.Image, rows: list[dict[str, str]]) -> None:
    draw = ImageDraw.Draw(image)
    x0 = WIDTH // 2
    left, right, bottom, top = x0 + 60, WIDTH - 20, HEIGHT - 60, 50
    small, title = font(14), font(16, bold=True)
    x_values = [float(row["tsne_1"]) for row in rows]
    y_values = [float(row["tsne_2"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span, y_span = max(x_max - x_min, 1e-8), max(y_max - y_min, 1e-8)
    draw.line((left, bottom, left, top), fill="#333333", width=2)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    centred(draw, ((left + right) / 2, HEIGHT - 25), "t-SNE dimension 1", small)
    draw.text((x0 + 2, (top + bottom) / 2 - 8), "t-SNE dimension 2", fill="#303030", font=small)
    groups = [
        ("Real normal", "#595959", "circle"),
        ("Pseudo-normal", "#2973c7", "circle"),
        ("Pseudo-abnormal", "#d64032", "square"),
    ]
    if {row["feature_group"] for row in rows} != {item[0] for item in groups}:
        raise ValueError("Projection contains unexpected or missing feature groups")
    for label, color, shape in groups:
        for row in rows:
            if row["feature_group"] != label:
                continue
            px = (float(row["tsne_1"]) - x_min) / x_span
            py = (float(row["tsne_2"]) - y_min) / y_span
            x = left + (0.02 + 0.96 * px) * (right - left)
            y = bottom - (0.02 + 0.96 * py) * (bottom - top)
            box = (x - 2, y - 2, x + 2, y + 2)
            draw.rectangle(box, fill=color) if shape == "square" else draw.ellipse(box, fill=color)
    legend_x = left
    for label, color, shape in groups:
        box = (legend_x, 28, legend_x + 7, 35)
        draw.rectangle(box, fill=color) if shape == "square" else draw.ellipse(box, fill=color)
        draw.text((legend_x + 13, 23), label, fill="#303030", font=small)
        legend_x += 13 + int(draw.textlength(label, font=small)) + 22
    draw.text((x0 + 7, 5), "(b) Prompt-conditioned feature projection", fill="#202020", font=title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the diagnostic JPEG from exported measurements")
    parser.add_argument(
        "--fusion", type=Path,
        default=ROOT / "code" / "materials" / "fusion_weight_trajectory.csv",
    )
    parser.add_argument(
        "--projection", type=Path,
        default=ROOT / "code" / "outputs" / "diagnostics" / "generated_feature_projection.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "latex" / "fusion_feature_analysis.jpg"
    )
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("The output must use the .jpg or .jpeg extension")
    fusion_rows, projection_rows = read_rows(args.fusion), read_rows(args.projection)
    require_columns(
        fusion_rows,
        {"stream_step", "lambda_text", "std_lambda_text", "lambda_history",
         "std_lambda_history", "lambda_reference", "std_lambda_reference"},
        args.fusion,
    )
    require_columns(
        projection_rows,
        {"feature_group", "point_index", "tsne_1", "tsne_2", "seed"},
        args.projection,
    )
    image = Image.new("RGBA", (WIDTH, HEIGHT), "white")
    line_plot(image, fusion_rows)
    projection_plot(image, projection_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(args.output, quality=95, subsampling=0)
    print(args.output)


if __name__ == "__main__":
    main()
