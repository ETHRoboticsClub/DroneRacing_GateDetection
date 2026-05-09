#!/usr/bin/env python3
"""Export the trained checkpoint to ONNX, build an FP16 TensorRT engine,
and report inference latency on whatever GPU we're running on.

Outputs the kind of numbers that answer 'will this fit on a Jetson Orin NX
at >=30 fps':
- ONNX export success / failure (the real risk for ASPP-based models)
- FP16 TRT engine build time
- N=200 batched-1 inference latency (mean / median / p99) on this GPU

Run from the repo root, in the gate-detection conda env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _flop_estimate(model: torch.nn.Module, shape: tuple[int, ...]) -> str:
    """Return a string with FLOP estimate, or '(thop not installed)'."""
    try:
        from thop import profile
        x = torch.zeros(*shape, device=next(model.parameters()).device)
        flops, params = profile(model, inputs=(x,), verbose=False)
        return f"{flops / 1e9:.2f} GFLOPs ({params/1e6:.1f} M params)"
    except Exception as e:                                   # noqa: BLE001
        return f"(thop unavailable: {e})"


def export_onnx(checkpoint: Path, onnx_path: Path, img_size: int = 384) -> tuple[int, str]:
    """Load checkpoint, unwrap the SegmentationModelWrapper, export to ONNX."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gate_detection.inference.predictor import load_checkpoint  # noqa: E402

    print(f"[export] loading {checkpoint}", flush=True)
    model, ckpt_meta = load_checkpoint(str(checkpoint), device="cuda")
    # SegmentationModelWrapper has a forward returning a tuple of heads.
    # ONNX needs a deterministic single output for downstream TRT/Isaac ROS;
    # take the last (full-resolution) head.
    base = model.model
    base.eval()

    flops_str = _flop_estimate(base, (1, 3, img_size, img_size))

    class FinalOutputOnly(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x):                                # noqa: D401
            y = self.m(x)
            if isinstance(y, (tuple, list)):
                return y[-1]
            return y

    wrapped = FinalOutputOnly(base).cuda().eval()

    dummy = torch.randn(1, 3, img_size, img_size, device="cuda")
    print(f"[export] flops: {flops_str}", flush=True)
    print(f"[export] writing {onnx_path}", flush=True)
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        opset_version=18,
        input_names=["input"],
        output_names=["mask"],
        dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}},
        do_constant_folding=True,
    )
    return onnx_path.stat().st_size, flops_str


def build_engine(onnx_path: Path, engine_path: Path, *, fp16: bool, max_workspace_gb: int = 4):
    """Build a TensorRT engine from ONNX. Returns (build_seconds, engine_bytes)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit-batch (default in TRT 10.x)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] parsing {onnx_path}", flush=True)
    # parse_from_file picks up the external-data .onnx.data file alongside.
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i)}", flush=True)
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled", flush=True)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 384, 384), (1, 3, 384, 384), (1, 3, 384, 384))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")
    build_s = time.perf_counter() - t0
    engine_bytes = bytes(serialized)
    engine_path.write_bytes(engine_bytes)
    return build_s, len(engine_bytes)


def benchmark(engine_path: Path, *, n_warmup: int = 20, n_iters: int = 200) -> dict:
    """Run TRT inference N times, return mean / median / p99 latency in ms."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    # IO bindings
    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    in_shape = (1, 3, 384, 384)
    out_shape = (1, 1, 384, 384)
    context.set_input_shape(in_name, in_shape)

    inp = torch.randn(*in_shape, dtype=torch.float32, device="cuda").contiguous()
    out = torch.empty(*out_shape, dtype=torch.float32, device="cuda").contiguous()
    context.set_tensor_address(in_name, int(inp.data_ptr()))
    context.set_tensor_address(out_name, int(out.data_ptr()))

    stream = torch.cuda.Stream()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

    # warm-up
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    # timed loop
    with torch.cuda.stream(stream):
        for i in range(n_iters):
            starts[i].record(stream)
            context.execute_async_v3(stream.cuda_stream)
            ends[i].record(stream)
    torch.cuda.synchronize()

    times_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return {
        "n":      int(n_iters),
        "mean":   float(times_ms.mean()),
        "median": float(np.median(times_ms)),
        "p99":    float(np.percentile(times_ms, 99)),
        "min":    float(times_ms.min()),
        "max":    float(times_ms.max()),
        "std":    float(times_ms.std()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_deeplabv3plus_imagenet_nonorm.pth")
    p.add_argument("--onnx",       default="checkpoints/gate_detection.onnx")
    p.add_argument("--engine",     default="checkpoints/gate_detection_fp16.engine")
    p.add_argument("--n-iters",    type=int, default=200)
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)

    onnx_path   = Path(args.onnx)
    engine_path = Path(args.engine)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    onnx_bytes, flops_str = export_onnx(Path(args.checkpoint), onnx_path)
    print(f"[export] OK — {onnx_bytes/1e6:.1f} MB ONNX", flush=True)

    build_s, eng_bytes = build_engine(onnx_path, engine_path, fp16=True)
    print(f"[build]  OK — {eng_bytes/1e6:.1f} MB engine, build {build_s:.1f} s", flush=True)

    bench = benchmark(engine_path, n_iters=args.n_iters)
    fps = 1000.0 / bench["mean"]
    print(f"\n[bench]  N={bench['n']} iters, FP16, {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  latency_mean   : {bench['mean']:6.2f} ms   ({fps:6.1f} fps)", flush=True)
    print(f"  latency_median : {bench['median']:6.2f} ms", flush=True)
    print(f"  latency_p99    : {bench['p99']:6.2f} ms", flush=True)
    print(f"  latency_min    : {bench['min']:6.2f} ms", flush=True)
    print(f"  latency_max    : {bench['max']:6.2f} ms", flush=True)
    print(f"  latency_std    : {bench['std']:6.2f} ms", flush=True)
    print(f"  flops          : {flops_str}", flush=True)
    print(f"\nProjection note: Jetson Orin NX 16 GB INT8 perf is ~50 INT8 TOPS ", flush=True)
    print(f"vs RTX 5090's ~3300 INT8 TOPS — i.e. Orin NX is ~65x slower at peak.", flush=True)
    print(f"With FP16 on Orin NX (~25 TOPS) the slowdown is ~10-30x in practice", flush=True)
    print(f"due to memory-bandwidth limits dominating over peak compute.", flush=True)


if __name__ == "__main__":
    main()
