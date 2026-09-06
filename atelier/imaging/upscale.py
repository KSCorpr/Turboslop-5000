"""Lossless metadata handling and context-aware tiled inference (no torch import)."""
from pathlib import Path
import os
import uuid

import numpy as np
from PIL import Image, ImageOps


def read_source(source):
    if isinstance(source, (str, Path)):
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).copy()
    else:
        image = ImageOps.exif_transpose(source).copy()
    return image


def save_result(rgb, source, output, expected):
    """Validate dimensions; preserve alpha and ICC without copying stale EXIF."""
    if rgb.size != tuple(expected):
        raise ValueError(f"Upscaler returned {rgb.size}; expected {tuple(expected)}. "
                         "No resize was applied to hide the mismatch.")
    result = rgb.convert("RGB")
    if "A" in source.getbands() or "transparency" in source.info:
        alpha = source.convert("RGBA").getchannel("A")
        result.putalpha(alpha.resize(expected, Image.Resampling.LANCZOS))
    output = Path(output)
    temp = output.with_name(output.name + f".{uuid.uuid4().hex}.part")
    try:
        result.save(temp, format="PNG", icc_profile=source.info.get("icc_profile"))
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return output


def tiled_x4(image, predict, tile=256, overlap=64, halo=32, log=None):
    """Blend overlapping predictions on CPU; only one padded tile reaches GPU.

    predict accepts RGB float32 HWC [0,1] and returns an exact 4x RGB array.
    The outer halo is discarded before feathering, to reduce border artifacts.
    """
    if tile <= overlap or overlap < 0 or halo < 0:
        raise ValueError("Invalid tile/overlap/halo")
    src = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    h, w = src.shape[:2]
    accum = np.zeros((h * 4, w * 4, 3), dtype=np.float32)
    weight = np.zeros((h * 4, w * 4, 1), dtype=np.float32)
    stride = tile - overlap
    ys = list(range(0, max(1, h - overlap), stride))
    xs = list(range(0, max(1, w - overlap), stride))
    count = 0
    for y in ys:
        for x in xs:
            y1, x1 = min(h, y + tile), min(w, x + tile)
            top, left = max(0, y - halo), max(0, x - halo)
            bottom, right = min(h, y1 + halo), min(w, x1 + halo)
            pred = np.asarray(predict(src[top:bottom, left:right]))
            if pred.shape != ((bottom-top)*4, (right-left)*4, 3):
                raise ValueError("The selected model must produce exactly 4x RGB.")
            if not np.isfinite(pred).all():
                raise ValueError("Model produced non-finite pixels; try full precision.")
            crop = pred[(y-top)*4:(y1-top)*4, (x-left)*4:(x1-left)*4]
            wy, wx = np.ones(crop.shape[0], np.float32), np.ones(crop.shape[1], np.float32)
            for window, start, end, limit in ((wy, y, y1, h), (wx, x, x1, w)):
                n = min(overlap * 4, len(window))
                if n:
                    ramp = np.arange(1, n+1, dtype=np.float32) / (n+1)
                    if start > 0:
                        window[:n] *= ramp
                    if end < limit:
                        window[-n:] *= ramp[::-1]
            mask = wy[:, None, None] * wx[None, :, None]
            accum[y*4:y1*4, x*4:x1*4] += crop * mask
            weight[y*4:y1*4, x*4:x1*4] += mask
            count += 1
            if log:
                log(f"Tile {count}/{len(xs)*len(ys)} ({tile}px)")
    if not (weight > 0).all():
        raise ValueError("Incomplete tile coverage")
    accum /= weight
    np.clip(accum, 0, 1, out=accum)
    accum *= 255
    np.rint(accum, out=accum)
    return Image.fromarray(accum.astype(np.uint8))
