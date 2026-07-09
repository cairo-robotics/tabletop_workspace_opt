#!/usr/bin/env python3
"""Generate printable AprilTag PNGs and a simple PDF contact sheet."""

import argparse
import math
import os

import cv2
from PIL import Image, ImageDraw, ImageFont


DICT_BY_FAMILY = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def _parse_ids(text):
    values = []
    for chunk in str(text).replace(",", " ").split():
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(chunk))
    return sorted(set(values))


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def generate_marker_image(dictionary, tag_id, marker_px, border_bits):
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_px, borderBits=border_bits)
    return Image.fromarray(marker).convert("L")


def build_printable_tile(marker_img, tag_id, family, marker_mm, dpi, page_margin_px):
    tile = Image.new("RGB", (marker_img.width + page_margin_px * 2, marker_img.height + page_margin_px * 2 + 80), "white")
    tile.paste(marker_img.convert("RGB"), (page_margin_px, page_margin_px))
    draw = ImageDraw.Draw(tile)
    title_font = _font(32)
    meta_font = _font(22)
    center_x = tile.width // 2
    title = f"{family}  id={tag_id}"
    meta = f"marker={marker_mm}mm  print={dpi}dpi"
    draw.text((center_x, marker_img.height + page_margin_px + 14), title, fill="black", anchor="ma", font=title_font)
    draw.text((center_x, marker_img.height + page_margin_px + 48), meta, fill="black", anchor="ma", font=meta_font)
    return tile


def save_contact_sheet(tiles, out_pdf, a4_dpi):
    page_w = int(round((210 / 25.4) * a4_dpi))
    page_h = int(round((297 / 25.4) * a4_dpi))
    margin = int(0.35 * a4_dpi)
    gap = int(0.18 * a4_dpi)
    cols = 2
    rows = 3
    slot_w = (page_w - 2 * margin - gap * (cols - 1)) // cols
    slot_h = (page_h - 2 * margin - gap * (rows - 1)) // rows

    pages = []
    per_page = cols * rows
    for page_idx in range(math.ceil(len(tiles) / float(per_page))):
        page = Image.new("RGB", (page_w, page_h), "white")
        for local_idx, tile in enumerate(tiles[page_idx * per_page:(page_idx + 1) * per_page]):
            row = local_idx // cols
            col = local_idx % cols
            x = margin + col * (slot_w + gap)
            y = margin + row * (slot_h + gap)
            scaled = tile.copy()
            scaled.thumbnail((slot_w, slot_h), Image.Resampling.LANCZOS)
            paste_x = x + (slot_w - scaled.width) // 2
            paste_y = y + (slot_h - scaled.height) // 2
            page.paste(scaled, (paste_x, paste_y))
        pages.append(page)

    pages[0].save(out_pdf, "PDF", resolution=a4_dpi, save_all=True, append_images=pages[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--ids", default="7-15")
    parser.add_argument("--marker-mm", type=float, default=60.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--border-bits", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        default="/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/assets/apriltags/tag36h11",
    )
    args = parser.parse_args()

    family = args.family.lower()
    if family not in DICT_BY_FAMILY:
        raise SystemExit(f"Unsupported family: {args.family}")
    os.makedirs(args.out_dir, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_BY_FAMILY[family])
    marker_px = int(round((args.marker_mm / 25.4) * args.dpi))
    ids = _parse_ids(args.ids)
    margin_px = int(round(0.18 * args.dpi))

    tiles = []
    for tag_id in ids:
        marker = generate_marker_image(dictionary, tag_id, marker_px, args.border_bits)
        tile = build_printable_tile(marker, tag_id, family, args.marker_mm, args.dpi, margin_px)
        png_path = os.path.join(args.out_dir, f"{family}_id_{tag_id:02d}_{int(args.marker_mm)}mm.png")
        tile.save(png_path, dpi=(args.dpi, args.dpi))
        tiles.append(tile)

    pdf_path = os.path.join(args.out_dir, f"{family}_ids_{ids[0]:02d}-{ids[-1]:02d}_{int(args.marker_mm)}mm_sheet.pdf")
    save_contact_sheet(tiles, pdf_path, args.dpi)
    print(args.out_dir)
    print(pdf_path)


if __name__ == "__main__":
    main()
