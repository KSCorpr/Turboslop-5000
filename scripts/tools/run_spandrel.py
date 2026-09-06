#!/usr/bin/env python3
"""Optional, prompt-free x4 inference. PyTorch stays outside the UI process."""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atelier.imaging.upscale import read_source, save_result, tiled_x4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--full-precision", action="store_true")
    args = ap.parse_args()
    import torch
    import numpy as np
    from spandrel import ModelLoader, ImageModelDescriptor
    from _device import pick_device

    device = pick_device(torch)
    model = ModelLoader().load_from_file(args.model)
    if not isinstance(model, ImageModelDescriptor):
        raise ValueError("Select an image super-resolution model.")
    if model.scale != 4 or model.input_channels != 3 or model.output_channels != 3:
        raise ValueError("Select a native x4 RGB model (DAT, HAT, ESRGAN…).")
    dtype = (torch.float16 if device == "cuda" and model.supports_half
             and not args.full_precision else torch.float32)
    model = model.to(device=device, dtype=dtype).eval()
    source = read_source(args.input)
    tile = 256
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        tile = 256 if free >= 6 * 1024**3 else 128
    print(f"{model.architecture.name}, x4, {device}, {dtype}, tile={tile}", flush=True)

    def predict(array):
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)
        with torch.inference_mode():
            result = model(tensor)
        return result[0].float().cpu().numpy().transpose(1, 2, 0)

    while True:
        try:
            result = tiled_x4(source, predict, tile=tile, overlap=32, halo=32,
                              log=lambda msg: print(msg, flush=True))
            break
        except torch.cuda.OutOfMemoryError:
            if device != "cuda" or tile <= 64:
                raise
            tile //= 2
        # Leave the exception scope before collecting tensors in its traceback.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory exhausted; retrying at {tile}px, still x4.", flush=True)
    save_result(result, source, args.output, (source.width*4, source.height*4))
    print(f"Saved {result.width}x{result.height}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
